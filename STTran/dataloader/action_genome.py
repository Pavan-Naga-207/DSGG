import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
try:
    import timm
    from timm.data import resolve_data_config
except Exception:
    timm = None
    resolve_data_config = None


_RESNET_PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]], dtype=np.float32)
_DEFAULT_VIT_DATA_CFG = {
    'mean': (0.5, 0.5, 0.5),
    'std': (0.5, 0.5, 0.5),
    'interpolation': 'bicubic',
    'input_size': (3, 224, 224),
}


def _round_up_to_multiple(value, divisor):
    if divisor <= 0:
        raise ValueError('divisor must be positive, got {}'.format(divisor))
    value = int(value)
    return int(np.ceil(float(value) / float(divisor)) * divisor)


def _parse_rgb_triplet(raw_value):
    values = [float(token.strip()) for token in str(raw_value).split(',') if token.strip()]
    if len(values) != 3:
        raise ValueError(
            'Expected an RGB triplet like "0.485,0.456,0.406", got {}'.format(raw_value)
        )
    return np.asarray(values, dtype=np.float32)


def _resolve_timm_data_cfg(model_name):
    if timm is None or resolve_data_config is None:
        return dict(_DEFAULT_VIT_DATA_CFG)

    backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    try:
        data_cfg = resolve_data_config({}, model=backbone)
    except TypeError:
        data_cfg = resolve_data_config(backbone.pretrained_cfg)
    resolved = dict(_DEFAULT_VIT_DATA_CFG)
    resolved.update(data_cfg)
    return resolved


def _cv2_interpolation_from_name(name):
    key = str(name).strip().lower()
    if key in ('bicubic', 'cubic'):
        return cv2.INTER_CUBIC
    if key in ('bilinear', 'linear'):
        return cv2.INTER_LINEAR
    if key in ('nearest',):
        return cv2.INTER_NEAREST
    if key in ('lanczos', 'lanczos4'):
        return cv2.INTER_LANCZOS4
    if key in ('area',):
        return cv2.INTER_AREA
    return cv2.INTER_CUBIC


def _resize_to_fit_square(image, target_size, interpolation):
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError('Invalid image shape: {}'.format(image.shape))
    scale = float(target_size) / float(max(height, width))
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    return resized, scale


def _pad_bottom_right(image, target_height, target_width, pad_value):
    channels = image.shape[2]
    canvas = np.empty((target_height, target_width, channels), dtype=image.dtype)
    canvas[...] = np.asarray(pad_value, dtype=image.dtype).reshape(1, 1, channels)
    canvas[: image.shape[0], : image.shape[1]] = image
    return canvas


def _build_detection_targets(gt_annotation, im_infos):
    frame_targets = []
    max_boxes = 0
    for frame_annotation, (_, _, im_scale) in zip(gt_annotation, im_infos):
        boxes = []
        if len(frame_annotation) > 0 and 'person_bbox' in frame_annotation[0]:
            person_boxes = np.asarray(frame_annotation[0]['person_bbox'], dtype=np.float32)
            if person_boxes.ndim == 1:
                person_boxes = person_boxes.reshape(1, -1)
            for bbox in person_boxes:
                if bbox.shape[0] >= 4:
                    boxes.append([bbox[0] * im_scale, bbox[1] * im_scale, bbox[2] * im_scale, bbox[3] * im_scale, 1.0])

        for rel in frame_annotation[1:]:
            bbox = np.asarray(rel['bbox'], dtype=np.float32).reshape(-1)
            if bbox.shape[0] < 4:
                continue
            boxes.append(
                [
                    bbox[0] * im_scale,
                    bbox[1] * im_scale,
                    bbox[2] * im_scale,
                    bbox[3] * im_scale,
                    float(rel['class']),
                ]
            )

        target = np.asarray(boxes, dtype=np.float32).reshape(-1, 5) if boxes else np.zeros((0, 5), dtype=np.float32)
        frame_targets.append(target)
        max_boxes = max(max_boxes, target.shape[0])

    padded_box_count = max(1, max_boxes)
    gt_boxes = torch.zeros((len(frame_targets), padded_box_count, 5), dtype=torch.float32)
    num_boxes = torch.zeros((len(frame_targets),), dtype=torch.int64)
    for frame_idx, target in enumerate(frame_targets):
        if target.shape[0] == 0:
            continue
        gt_boxes[frame_idx, : target.shape[0]] = torch.from_numpy(target)
        num_boxes[frame_idx] = int(target.shape[0])
    return gt_boxes, num_boxes

class AG(Dataset):

    def __init__(
        self,
        mode,
        datasize,
        data_path=None,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
        backbone='resnet101',
        return_detection_targets=False,
        min_frames_per_video=3,
    ):

        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')
        self.backbone = str(backbone).lower()
        self.return_detection_targets = bool(return_detection_targets)
        self.min_frames_per_video = int(min_frames_per_video)
        if self.backbone not in ('resnet101', 'vitdet'):
            raise ValueError(
                'Unsupported backbone {}. Expected resnet101 or vitdet.'.format(backbone)
            )
        self.vit_patch_size = int(os.environ.get('VIT_PATCH_SIZE', '16'))
        self.vit_input_size = _round_up_to_multiple(
            int(os.environ.get('VIT_INPUT_SIZE', '1024')),
            self.vit_patch_size,
        )
        self.vit_model_name = os.environ.get('VITDET_MODEL', 'vit_base_patch16_224')
        self.vit_data_cfg = _resolve_timm_data_cfg(self.vit_model_name)
        default_pad_rgb = ','.join(str(value) for value in self.vit_data_cfg['mean'])
        self.vit_pad_rgb = _parse_rgb_triplet(
            os.environ.get('VIT_PAD_RGB', default_pad_rgb)
        )
        self.vit_interpolation_name = os.environ.get(
            'VIT_INTERPOLATION',
            self.vit_data_cfg.get('interpolation', 'bicubic'),
        )
        self.vit_interpolation = _cv2_interpolation_from_name(self.vit_interpolation_name)

        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]


        print('-------loading annotations---------slowly-----------')

        if filter_small_box:
            with open(os.path.join(root_path, 'annotations/person_bbox.pkl'), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open('dataloader/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(os.path.join(root_path, 'annotations/person_bbox.pkl'), 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(os.path.join(root_path, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()
        print('--------------------finish!-------------------------')

        if datasize == 'mini':
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:80000]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object


        # collect valid frames
        video_dict = {}
        for i in person_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is visible bbox
                    if j['visible']:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]

        self.video_list = []
        self.video_size = [] # (w,h)
        self.gt_annotations = []
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.short_video = 0
        self.valid_nums = 0

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    if person_bbox[j]['bbox'].shape[0] == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self.valid_nums += 1


                gt_annotation_frame = [{'person_bbox': person_bbox[j]['bbox']}]
                # each frames's objects and human
                for k in object_bbox[j]:
                    if k['visible']:
                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
                        k['class'] = self.object_classes.index(k['class'])
                        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
                        k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
                        k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
                        k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            if len(video) >= self.min_frames_per_video:
                self.video_list.append(video)
                self.video_size.append(person_bbox[j]['bbox_size'])
                self.gt_annotations.append(gt_annotation_video)
            elif len(video) > 0:
                self.short_video += 1
            else:
                self.non_person_video += 1

        print('x'*60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            print(
                '{} videos are invalid (<{} usable frames), remove them'.format(
                    self.short_video,
                    self.min_frames_per_video,
                )
            )
            print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(self.non_heatmap_nums))
        print('x' * 60)
        if self.backbone == 'vitdet':
            print(
                'ViT input pipeline: RGB float[0,1], fit-longest-side={}, square-pad={}, patch={}'.format(
                    self.vit_input_size,
                    self.vit_input_size,
                    self.vit_patch_size,
                )
            )
            print(
                'ViT data cfg: model={} interpolation={} mean={} std={}'.format(
                    self.vit_model_name,
                    self.vit_interpolation_name,
                    tuple(self.vit_data_cfg.get('mean', ())),
                    tuple(self.vit_data_cfg.get('std', ())),
                )
            )

    def __getitem__(self, index):

        frame_names = self.video_list[index]
        processed_ims = []
        im_infos = []

        for name in frame_names:
            frame_path = os.path.join(self.frames_path, name)
            im = cv2.imread(frame_path, cv2.IMREAD_COLOR) # bgr, h,w,3
            if im is None:
                raise FileNotFoundError('Failed to read frame: {}'.format(frame_path))

            if self.backbone == 'vitdet':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = im.astype(np.float32) / 255.0
                resized_im, im_scale = _resize_to_fit_square(
                    im,
                    self.vit_input_size,
                    self.vit_interpolation,
                )
                resized_h, resized_w = resized_im.shape[:2]
                padded_im = _pad_bottom_right(
                    resized_im,
                    self.vit_input_size,
                    self.vit_input_size,
                    self.vit_pad_rgb,
                )
                processed_ims.append(np.ascontiguousarray(padded_im))
                im_infos.append([resized_h, resized_w, im_scale])
            else:
                im, im_scale = prep_im_for_blob(
                    im,
                    _RESNET_PIXEL_MEANS,
                    600,
                    1000,
                ) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
                processed_ims.append(im)
                im_infos.append([im.shape[0], im.shape[1], im_scale])

        if self.backbone == 'vitdet':
            blob = np.stack(processed_ims, axis=0)
        else:
            blob = im_list_to_blob(processed_ims)

        im_info = torch.from_numpy(np.asarray(im_infos, dtype=np.float32))
        img_tensor = torch.from_numpy(np.ascontiguousarray(blob))
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        if self.return_detection_targets:
            gt_boxes, num_boxes = _build_detection_targets(self.gt_annotations[index], im_infos)
        else:
            gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
            num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index

    def __len__(self):
        return len(self.video_list)

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]
