import torch
from torch import nn
import math
from model_utils import *
from block import fusions
from collections import OrderedDict


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view((-1,)+self.shape)


class PositionalEncoding(nn.Module):

    def __init__(self, feat_size, dropout=0.1, max_len=4):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, feat_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, feat_size, 2).float() * (-math.log(10000.0) / feat_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape, self.pe.shape)
        x = x + self.pe
        # return self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, feat_size, hidden_size=256, nhead=4, num_encoder_layers=3, max_len=4, num_decoder_layers=-1, num_queries=4, spatial_dim=-1):
        super(Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(feat_size, max_len=max_len)
        encoder_layers = nn.TransformerEncoderLayer(
            feat_size, nhead, hidden_size)

        self.spatial_dim = spatial_dim
        if self.spatial_dim != -1:
            transformer_encoder_spatial_layers = nn.TransformerEncoderLayer(
                spatial_dim, nhead, hidden_size)
            self.transformer_encoder_spatial = nn.TransformerEncoder(
                transformer_encoder_spatial_layers, num_encoder_layers)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_encoder_layers)
        self.use_decoder = (num_decoder_layers != -1)

        if self.use_decoder:
            decoder_layers = nn.TransformerDecoderLayer(
                hidden_size, nhead, hidden_size)
            self.transformer_decoder = nn.TransformerDecoder(
                decoder_layers, num_decoder_layers, norm=nn.LayerNorm(hidden_size))
            self.tgt_pos = nn.Embedding(num_queries, hidden_size).weight
            assert self.tgt_pos.requires_grad == True

    def forward(self, embeddings, idx):
        ''' embeddings: CxBxCh*H*W '''
        # print(embeddings.shape)
        batch_size = embeddings.size(1)

        if self.spatial_dim != -1:
            embeddings = embeddings.permute((2, 1, 0))
            embeddings = self.transformer_encoder_spatial(embeddings)
            embeddings = embeddings.permute((2, 1, 0))

        x = self.pos_encoder(embeddings)
        x = self.transformer_encoder(x)
        if self.use_decoder:
            if idx != -1:
                tgt_pos = self.tgt_pos[idx].unsqueeze(0)
                # print(tgt_pos.size())
                tgt_pos = tgt_pos.unsqueeze(1).repeat(1, batch_size, 1)
            else:
                tgt_pos = self.tgt_pos.unsqueeze(1).repeat(1, batch_size, 1)
            tgt = torch.zeros_like(tgt_pos)
            x = self.transformer_decoder(tgt + tgt_pos, x)
        return x


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes,
                 use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.inplanes = inplanes
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)

        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)

        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)

        out = torch.cat([out1, out2], 1)
        if self.use_bypass:
            out += x
        out = self.relu(out)

        return out


class SqueezeNet(nn.Module):

    def __init__(self,
                 version=1.1,
                 num_classes=600):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes

        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv3d(3, 96, kernel_size=7, stride=(
                    1, 2, 2), padding=(3, 3, 3)),
                nn.BatchNorm3d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192, use_bypass=True),
                Fire(384, 64, 256, 256),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(512, 64, 256, 256, use_bypass=True),
            )
        if version == 1.1:
            self.features = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3, stride=(
                    1, 2, 2), padding=(1, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256, use_bypass=True),
            )
            self.module1 = self.features[:6]
            self.module2 = self.features[6:9]
            self.module3 = self.features[9:12]
            self.module4 = self.features[12:]
        # Final convolution is initialized differently form the rest
        # final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1)
        # self.classifier = nn.Sequential(
        # 	nn.Dropout(p=0.5),
        # 	final_conv,
        # 	nn.ReLU(inplace=True),
        # 	nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # print(x.size())
        # y = self.features[0](x)
        # for i in range(1, len(self.features)):
        #     y = self.features[i](y)
        #     print(i, y.size())

        y3 = self.module1(x)
        y2 = self.module2(y3)
        y1 = self.module3(y2)
        y0 = self.module4(y1)

        # x = self.features(x)
        # print(x.size(), y0.size(), y1.size(), y2.size(), y3.size())

        return [y0, y1, y2, y3]


class VideoSaliencyModel(nn.Module):
    def __init__(self,
                 img_backbone='s3d',
                 transformer_in_channel=32,
                 nhead=4,
                 use_upsample=True,
                 num_hier=3,
                 num_clips=32
                 ):
        super(VideoSaliencyModel, self).__init__()

        self.img_backbone = img_backbone
        if img_backbone == 's3d':
            self.backbone = BackBoneS3D()
        elif img_backbone == 'squeezenet':
            self.backbone = SqueezeNet()
        else:
            raise NotImplementedError()

        self.num_hier = num_hier
        if use_upsample:
            if num_hier == 0:
                self.decoder = DecoderConvUpNoHier(img_backbone=img_backbone)
            elif num_hier == 1:
                self.decoder = DecoderConvUp1Hier(img_backbone=img_backbone)
            elif num_hier == 2:
                self.decoder = DecoderConvUp2Hier(img_backbone=img_backbone)
            elif num_hier == 3:
                if num_clips == 8:
                    self.decoder = DecoderConvUp8(img_backbone=img_backbone)
                elif num_clips == 16:
                    self.decoder = DecoderConvUp16(img_backbone=img_backbone)
                elif num_clips == 32:
                    self.decoder = DecoderConvUp(img_backbone=img_backbone)
                elif num_clips == 48:
                    self.decoder = DecoderConvUp48(img_backbone=img_backbone)
        else:
            self.decoder = DecoderConvT()

    def forward(self, x):
        [y0, y1, y2, y3] = self.backbone(x)
        if self.num_hier == 0:
            return self.decoder(y0)
        if self.num_hier == 1:
            return self.decoder(y0, y1)
        if self.num_hier == 2:
            return self.decoder(y0, y1, y2)
        if self.num_hier == 3:
            return self.decoder(y0, y1, y2, y3)


class VideoAudioSaliencyFusionModel(nn.Module):
    def __init__(self,
                 use_transformer=True,
                 transformer_in_channel=512,
                 num_encoder_layers=3,
                 nhead=4,
                 use_upsample=True,
                 num_hier=3,
                 num_clips=32
                 ):
        super(VideoAudioSaliencyFusionModel, self).__init__()
        self.use_transformer = use_transformer
        self.visual_model = VideoSaliencyModel(
            transformer_in_channel=transformer_in_channel,
            nhead=nhead,
            use_upsample=use_upsample,
            num_hier=num_hier,
            num_clips=num_clips
        )

        self.conv_in_1x1 = nn.Conv3d(
            in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
        self.transformer = Transformer(
            transformer_in_channel,
            hidden_size=transformer_in_channel,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=-1,
            max_len=4*7*12+3,
        )

        self.audionet = SoundNet()
        self.audio_conv_1x1 = nn.Conv2d(
            in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
        self.audionet.load_state_dict(torch.load('./soundnet8_final.pth'))
        print("Loaded SoundNet Weights")
        for param in self.audionet.parameters():
            param.requires_grad = True

        self.maxpool = nn.MaxPool3d(
            (4, 1, 1), stride=(2, 1, 2), padding=(0, 0, 0))
        self.bilinear = nn.Bilinear(42, 3, 4*7*12)

    def forward(self, x, audio):
        audio = self.audionet(audio)
        # print(audio.size())
        audio = self.audio_conv_1x1(audio)
        audio = audio.flatten(2)
        # print("audio", audio.shape)

        [y0, y1, y2, y3] = self.visual_model.backbone(x)
        y0 = self.conv_in_1x1(y0)
        y0 = y0.flatten(2)
        # print("video", y0.shape)

        fused_out = torch.cat((y0, audio), 2)
        # print("fused_out", fused_out.size())
        fused_out = fused_out.permute((2, 0, 1))
        fused_out = self.transformer(fused_out, -1)

        fused_out = fused_out.permute((1, 2, 0))

        video_features = fused_out[..., :4*7*12]
        audio_features = fused_out[..., 4*7*12:]

        # print("separate", video_features.shape, audio_features.shape)

        video_features = video_features.view(
            video_features.size(0), video_features.size(1), 4, 7, 12)
        audio_features = torch.mean(audio_features, dim=2)

        audio_features = audio_features.view(audio_features.size(
            0), audio_features.size(1), 1, 1, 1).repeat(1, 1, 4, 7, 12)

        final_out = torch.cat((video_features, audio_features), 1)

        # print(final_out.size())

        return self.visual_model.decoder(final_out, y1, y2, y3)


class VideoAudioSaliencyModel(nn.Module):
    def __init__(self,
                 img_backbone='s3d',
                 use_transformer=False,
                 transformer_in_channel=32,
                 num_encoder_layers=3,
                 nhead=4,
                 use_upsample=True,
                 num_hier=3,
                 num_clips=32
                 ):
        super(VideoAudioSaliencyModel, self).__init__()
        self.use_transformer = use_transformer
        self.visual_model = VideoSaliencyModel(
            img_backbone=img_backbone,
            transformer_in_channel=transformer_in_channel,
            nhead=nhead,
            use_upsample=use_upsample,
            num_hier=num_hier,
            num_clips=num_clips
        )

        if self.use_transformer:
            self.conv_in_1x1 = nn.Conv3d(
                in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1, bias=True)
            self.conv_out_1x1 = nn.Conv3d(
                in_channels=32, out_channels=1024, kernel_size=1, stride=1, bias=True)
            self.transformer = Transformer(
                4*7*12,
                hidden_size=4*7*12,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=-1,
                max_len=transformer_in_channel,
            )

        self.audionet = SoundNet()
        self.audionet.load_state_dict(torch.load('./soundnet8_final.pth'))
        print("Loaded SoundNet Weights")
        for param in self.audionet.parameters():
            param.requires_grad = True

        self.maxpool = nn.MaxPool3d(
            (4, 1, 1), stride=(2, 1, 2), padding=(0, 0, 0))
        self.bilinear = nn.Bilinear(42, 3, 4*7*12)

    def forward(self, x, audio):
        audio = self.audionet(audio)
        [y0, y1, y2, y3] = self.visual_model.backbone(x)
        y0 = self.maxpool(y0)
        fused_out = self.bilinear(y0.flatten(2), audio.flatten(2))
        fused_out = fused_out.view(
            fused_out.size(0), fused_out.size(1), 4, 7, 12)

        if self.use_transformer:
            fused_out = self.conv_in_1x1(fused_out)
            fused_out = fused_out.flatten(2)
            fused_out = fused_out.permute((1, 0, 2))
            # print("fused_out", fused_out.shape)
            fused_out = self.transformer(fused_out, -1)
            fused_out = fused_out.permute((1, 0, 2))
            fused_out = fused_out.view(
                fused_out.size(0), fused_out.size(1), 4, 7, 12)
            fused_out = self.conv_out_1x1(fused_out)

        return self.visual_model.decoder(fused_out, y1, y2, y3)


class DecoderConvUp(nn.Module):
    def __init__(self, img_backbone):
        super(DecoderConvUp, self).__init__()
        if img_backbone == 's3d':
            lists = [1024, 832, 480, 192, 64, 32, 1]
            self.upsampling = nn.Upsample(
                scale_factor=(1, 2, 2), mode='trilinear')
            self.single_upsampling = nn.Upsample(
                scale_factor=(1, 2, 2), mode='trilinear')
        elif img_backbone == 'squeezenet':
            lists = [512, 384, 256, 128, 64, 32, 1]
            self.upsampling = nn.Upsample(
                scale_factor=(2, 2, 2), mode='trilinear')
            self.single_upsampling = nn.Upsample(
                scale_factor=(1, 2, 2), mode='trilinear')

        self.convtsp1 = nn.Sequential(
            nn.Conv3d(lists[0], lists[1], kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(lists[1], lists[2], kernel_size=(3, 3, 3), stride=(
                3, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(lists[2], lists[3], kernel_size=(5, 3, 3), stride=(
                5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(lists[3], lists[4], kernel_size=(5, 3, 3), stride=(
                5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.single_upsampling,  # 112 x 192

            nn.Conv3d(lists[4], lists[5], kernel_size=(2, 3, 3), stride=(
                2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.single_upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv3d(lists[5], lists[5], kernel_size=(2, 1, 1),
                      stride=(2, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(lists[5], lists[6], kernel_size=1,
                      stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, y0, y1, y2, y3):
        z = self.convtsp1(y0)
        # print('convtsp1', z.shape)

        z = torch.cat((z, y1), 2)
        # print('cat_convtsp1', z.shape)

        z = self.convtsp2(z)
        # print('convtsp2', z.shape)

        z = torch.cat((z, y2), 2)
        # print('cat_convtsp2', z.shape)

        z = self.convtsp3(z)
        # print('convtsp3', z.shape)

        z = torch.cat((z, y3), 2)
        # print("cat_convtsp3", z.shape)

        z = self.convtsp4(z)
        # print('convtsp4', z.shape)

        z = z.view(z.size(0), z.size(3), z.size(4))
        # print('output', z.shape)

        return z


class DecoderConvUp16(nn.Module):
    def __init__(self, img_backbone):
        super(DecoderConvUp16, self).__init__()
        if img_backbone == 's3d':
            lists = [1024, 832, 480, 192, 64, 32, 1]
        elif img_backbone == 'squeezenet':
            lists = [512, 384, 256, 128, 64, 32, 1]
        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(lists[0], lists[1], kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(lists[1], lists[2], kernel_size=(3, 3, 3), stride=(
                3, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(lists[2], lists[3], kernel_size=(5, 3, 3), stride=(
                5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(lists[3], lists[4], kernel_size=(5, 3, 3), stride=(
                5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 112 x 192

            nn.Conv3d(lists[4], lists[5], kernel_size=(2, 3, 3), stride=(
                2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv3d(lists[5], lists[6], kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), bias=True),
            # nn.ReLU(),
            # nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, y0, y1, y2, y3):
        z = self.convtsp1(y0)
        # print('convtsp1', z.shape)

        z = torch.cat((z, y1), 2)
        # print('cat_convtsp1', z.shape)

        z = self.convtsp2(z)
        # print('convtsp2', z.shape)

        z = torch.cat((z, y2), 2)
        # print('cat_convtsp2', z.shape)

        z = self.convtsp3(z)
        # print('convtsp3', z.shape)

        z = torch.cat((z, y3), 2)
        # print("cat_convtsp3", z.shape)

        z = self.convtsp4(z)
        # print('convtsp4', z.shape)

        z = z.view(z.size(0), z.size(3), z.size(4))
        # print('output', z.shape)

        return z


class DecoderConvUp8(nn.Module):
    def __init__(self, img_backbone):
        super(DecoderConvUp8, self).__init__()
        if img_backbone == 's3d':
            lists = [1024, 832, 480, 192, 64, 32, 1]
        elif img_backbone == 'squeezenet':
            lists = [512, 384, 256, 128, 64, 32, 1]
        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(lists[0], lists[1], kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(lists[1], lists[2], kernel_size=(3, 3, 3), stride=(
                3, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(lists[2], lists[3], kernel_size=(5, 3, 3), stride=(
                5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(lists[3], lists[4], kernel_size=(5, 3, 3), stride=(
                5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 112 x 192

            nn.Conv3d(lists[4], lists[5], kernel_size=(1, 3, 3), stride=(
                1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv3d(lists[5], lists[6], kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), bias=True),
            # nn.ReLU(),
            # nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, y0, y1, y2, y3):
        z = self.convtsp1(y0)
        # print('convtsp1', z.shape)

        z = torch.cat((z, y1), 2)
        # print('cat_convtsp1', z.shape)

        z = self.convtsp2(z)
        # print('convtsp2', z.shape)

        z = torch.cat((z, y2), 2)
        # print('cat_convtsp2', z.shape)

        z = self.convtsp3(z)
        # print('convtsp3', z.shape)

        z = torch.cat((z, y3), 2)
        # print("cat_convtsp3", z.shape)

        z = self.convtsp4(z)
        # print('convtsp4', z.shape)

        z = z.view(z.size(0), z.size(3), z.size(4))
        # print('output', z.shape)

        return z


class DecoderConvUp48(nn.Module):
    def __init__(self, img_backbone):
        super(DecoderConvUp48, self).__init__()
        if img_backbone == 's3d':
            lists = [1024, 832, 480, 192, 64, 32, 1]
        elif img_backbone == 'squeezenet':
            lists = [512, 384, 256, 128, 64, 32, 1]
        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(lists[0], lists[1], kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(lists[1], lists[2], kernel_size=(3, 3, 3), stride=(
                3, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(lists[2], lists[3], kernel_size=(5, 3, 3), stride=(
                5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(lists[3], lists[4], kernel_size=(5, 3, 3), stride=(
                5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 112 x 192

            nn.Conv3d(lists[4], lists[5], kernel_size=(2, 3, 3), stride=(
                2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv3d(lists[5], lists[5], kernel_size=(3, 1, 1),
                      stride=(3, 1, 1), bias=True),
            nn.ReLU(),
            nn.Conv3d(lists[5], lists[6], kernel_size=1,
                      stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, y0, y1, y2, y3):
        # print(y0.shape)
        z = self.convtsp1(y0)
        # print('convtsp1', z.shape)

        z = torch.cat((z, y1), 2)
        # print('cat_convtsp1', z.shape)

        z = self.convtsp2(z)
        # print('convtsp2', z.shape)

        z = torch.cat((z, y2), 2)
        # print('cat_convtsp2', z.shape)

        z = self.convtsp3(z)
        # print('convtsp3', z.shape)

        z = torch.cat((z, y3), 2)
        # print("cat_convtsp3", z.shape)

        z = self.convtsp4(z)
        # print('convtsp4', z.shape)

        z = z.view(z.size(0), z.size(3), z.size(4))
        # print('output', z.shape)

        return z


class DecoderConvUpNoHier(nn.Module):
    def __init__(self, img_backbone):
        super(DecoderConvUpNoHier, self).__init__()
        if img_backbone == 's3d':
            lists = [1024, 832, 480, 192, 64, 32, 1]
        elif img_backbone == 'squeezenet':
            lists = [512, 384, 256, 128, 64, 32, 1]
        elif img_backbone == 'tsm':
            lists = [512, 384, 256, 128, 64, 32, 1]
        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(lists[0], lists[1], kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(lists[1], lists[2], kernel_size=(1, 3, 3), stride=(
                1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(lists[2], lists[3], kernel_size=(1, 3, 3), stride=(
                1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(lists[3], lists[4], kernel_size=(1, 3, 3), stride=(
                1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 112 x 192

            nn.Conv3d(lists[4], lists[5], kernel_size=(2, 3, 3), stride=(
                2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv3d(lists[5], lists[5], kernel_size=(2, 1, 1),
                      stride=(2, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(lists[5], lists[6], kernel_size=1,
                      stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, y0):

        z = self.convtsp1(y0)
        # print('convtsp1', z.shape)

        # z = torch.cat((z,y1), 2)
        # print('cat_convtsp1', z.shape)

        z = self.convtsp2(z)
        # print('convtsp2', z.shape)

        # z = torch.cat((z,y2), 2)
        # print('cat_convtsp2', z.shape)

        z = self.convtsp3(z)
        # print('convtsp3', z.shape)

        # z = torch.cat((z,y3), 2)
        # print("cat_convtsp3", z.shape)

        z = self.convtsp4(z)
        # print('convtsp4', z.shape)

        z = z.view(z.size(0), z.size(3), z.size(4))
        # print('output', z.shape)

        return z


class DecoderConvUp1Hier(nn.Module):
    def __init__(self, img_backbone):
        super(DecoderConvUp1Hier, self).__init__()
        if img_backbone == 's3d':
            lists = [1024, 832, 480, 192, 64, 32, 1]
        elif img_backbone == 'squeezenet':
            lists = [512, 384, 256, 128, 64, 32, 1]
        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(lists[0], lists[1], kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(lists[1], lists[2], kernel_size=(3, 3, 3), stride=(
                3, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(lists[2], lists[3], kernel_size=(1, 3, 3), stride=(
                1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(lists[3], lists[4], kernel_size=(1, 3, 3), stride=(
                1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 112 x 192

            nn.Conv3d(lists[4], lists[5], kernel_size=(2, 3, 3), stride=(
                2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv3d(lists[5], lists[5], kernel_size=(2, 1, 1),
                      stride=(2, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(lists[5], lists[6], kernel_size=1,
                      stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, y0, y1):

        z = self.convtsp1(y0)
        # print('convtsp1', z.shape, y1.shape)

        z = torch.cat((z, y1), 2)
        # print('cat_convtsp1', z.shape)

        z = self.convtsp2(z)
        # print('convtsp2', z.shape)

        # z = torch.cat((z,y2), 2)
        # print('cat_convtsp2', z.shape)

        z = self.convtsp3(z)
        # print('convtsp3', z.shape)

        # z = torch.cat((z,y3), 2)
        # print("cat_convtsp3", z.shape)

        z = self.convtsp4(z)
        # print('convtsp4', z.shape)

        z = z.view(z.size(0), z.size(3), z.size(4))
        # print('output', z.shape)

        return z


class DecoderConvUp2Hier(nn.Module):
    def __init__(self, img_backbone):
        super(DecoderConvUp2Hier, self).__init__()
        if img_backbone == 's3d':
            lists = [1024, 832, 480, 192, 64, 32, 1]
        elif img_backbone == 'squeezenet':
            lists = [512, 384, 256, 128, 64, 32, 1]
        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(lists[0], lists[1], kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(lists[1], lists[2], kernel_size=(3, 3, 3), stride=(
                3, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(lists[2], lists[3], kernel_size=(5, 3, 3), stride=(
                5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(lists[3], lists[4], kernel_size=(1, 3, 3), stride=(
                1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 112 x 192

            nn.Conv3d(lists[4], lists[5], kernel_size=(2, 3, 3), stride=(
                2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv3d(lists[5], lists[5], kernel_size=(2, 1, 1),
                      stride=(2, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(lists[5], lists[6], kernel_size=1,
                      stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, y0, y1, y2):

        z = self.convtsp1(y0)
        # print('convtsp1', z.shape)

        z = torch.cat((z, y1), 2)
        # print('cat_convtsp1', z.shape)

        z = self.convtsp2(z)
        # print('convtsp2', z.shape)

        z = torch.cat((z, y2), 2)
        # print('cat_convtsp2', z.shape)

        z = self.convtsp3(z)
        # print('convtsp3', z.shape)

        # z = torch.cat((z,y3), 2)
        # print("cat_convtsp3", z.shape)

        z = self.convtsp4(z)
        # print('convtsp4', z.shape)

        z = z.view(z.size(0), z.size(3), z.size(4))
        # print('output', z.shape)

        return z


class BackBoneS3D(nn.Module):
    def __init__(self):
        super(BackBoneS3D, self).__init__()

        self.base1 = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(
                1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
        )
        self.maxp2 = nn.MaxPool3d(kernel_size=(
            1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.base2 = nn.Sequential(
            Mixed_3b(),
            Mixed_3c(),
        )
        self.maxp3 = nn.MaxPool3d(kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.base3 = nn.Sequential(
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
        )
        self.maxt4 = nn.MaxPool3d(kernel_size=(
            2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.maxp4 = nn.MaxPool3d(kernel_size=(
            1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.base4 = nn.Sequential(
            Mixed_5b(),
            Mixed_5c(),
        )

    def forward(self, x):
        # print('input', x.shape)
        y3 = self.base1(x)
        # print('base1', y3.shape)

        y = self.maxp2(y3)
        # print('maxp2', y.shape)

        y2 = self.base2(y)
        # print('base2', y2.shape)

        y = self.maxp3(y2)
        # print('maxp3', y.shape)

        y1 = self.base3(y)
        # print('base3', y1.shape)

        y = self.maxt4(y1)
        y = self.maxp4(y)
        # print('maxt4p4', y.shape)

        y0 = self.base4(y)
        # print(x.size(), y0.size(), y1.size(), y2.size(), y3.size())

        return [y0, y1, y2, y3]


class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1),
                               padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
                               padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
                               padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1),
                                    stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1),
                                    stride=(2, 1))

    def forward(self, waveform):
        x = self.conv1(waveform)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)

        return x


class DecoderConv2D(nn.Module):
    def __init__(self, base_model):
        super(DecoderConv2D, self).__init__()
        if 'resnet' in base_model:
            lists = [2048, 1024, 512, 256, 64, 32, 1]
        elif base_model == 'mobilenetv2':
            lists = [1280, 160, 64, 32, 24, 16, 1]

        self.upsampling = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.convtsp1 = nn.Sequential(
            nn.Conv2d(lists[0], lists[1], kernel_size=(3, 3),
                      stride=1, padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv2d(lists[1], lists[2], kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv2d(lists[2], lists[3], kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv2d(lists[3], lists[4], kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 112 x 192

            nn.Conv2d(lists[4], lists[5], kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv2d(lists[5], lists[5], kernel_size=(1, 1),
                      stride=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(lists[5], lists[6], kernel_size=1,
                      stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, y0):

        # import pdb
        # pdb.set_trace()

        z = self.convtsp1(y0)
        # print('convtsp1', z.shape)

        # z = torch.cat((z,y1), 2)
        # print('cat_convtsp1', z.shape)

        z = self.convtsp2(z)
        # print('convtsp2', z.shape)

        # z = torch.cat((z,y2), 2)
        # print('cat_convtsp2', z.shape)

        z = self.convtsp3(z)
        # print('convtsp3', z.shape)

        # z = torch.cat((z,y3), 2)
        # print("cat_convtsp3", z.shape)

        z = self.convtsp4(z)
        # print('convtsp4', z.shape)

        z = z.view(z.size(0), z.size(2), z.size(3))
        # print('output', z.shape)

        return z


class DecoderConv2D3hier(nn.Module):
    def __init__(self, base_model):
        super(DecoderConv2D3hier, self).__init__()
        if 'resnet' in base_model:
            lists = [2048, 1024, 512, 256, 64, 32, 1]

            self.upsampling = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
            self.single_upsampling = nn.Upsample(
                scale_factor=(2, 2), mode='bilinear')

        elif base_model == 'mobilenetv2':
            lists = [1280, 160, 64, 32, 24, 16, 1]

            self.upsampling = nn.Upsample(
                scale_factor=(2, 2), mode='bilinear')

        self.convtsp1 = nn.Sequential(
            nn.Conv2d(lists[0], lists[1], kernel_size=(3, 3),
                      stride=1, padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv2d(lists[1]*2, lists[2], kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv2d(lists[2]*2, lists[3], kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv2d(lists[3]*2, lists[4], kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 112 x 192

            nn.Conv2d(lists[4], lists[5], kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv2d(lists[5], lists[5], kernel_size=(1, 1),
                      stride=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(lists[5], lists[6], kernel_size=1,
                      stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, y0, y1, y2, y3):

        # import pdb
        # pdb.set_trace()

        # [y0, y1, y2, y3] = y_list

        z = self.convtsp1(y0)
        # print('convtsp1', z.shape)

        y1 = self.upsampling(y1)

        z = torch.cat((z, y1), 1)
        # print('cat_convtsp1', z.shape)

        z = self.convtsp2(z)
        # print('convtsp2', z.shape)

        y2 = self.upsampling(y2)

        z = torch.cat((z, y2), 1)
        # print('cat_convtsp2', z.shape)

        z = self.convtsp3(z)
        # print('convtsp3', z.shape)

        y3 = self.upsampling(y3)

        z = torch.cat((z, y3), 1)
        # print("cat_convtsp3", z.shape)

        z = self.convtsp4(z)
        # print('convtsp4', z.shape)

        z = z.view(z.size(0), z.size(2), z.size(3))
        # print('output', z.shape)

        return z
