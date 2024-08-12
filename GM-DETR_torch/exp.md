Dataset for two-ways input
```bash
class CocoXDetection(torchvision.datasets.VisionDataset):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False):
        super().__init__(img_folder, transforms)
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
    
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")
    
    def _load_image_X(self, id: int) -> Image.Image:
        path_0 = self.coco.loadImgs(id)[0]["file_name"]
        path_1 = path_0.replace('data_ir', 'data_rgb')
        path_1 = path_1.replace('jpeg', 'jpg')
        path_1 = path_1.replace('PreviewData', 'RGB')
        image_0 = Image.open(os.path.join(self.root, path_0)).convert("RGB")
        image_1 = Image.open(os.path.join(self.root, path_1)).convert("RGB")
        return image_0, image_1
    
    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def getitem_X(self, index: int) -> Tuple[Any, Any, Any]:
        id = self.ids[index]
        image_0, image_1 = self._load_image_X(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image_0, image_1, target
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx):
        img0, img1, target = self.getitem_X(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img0, target = self.prepare(img0, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=img0.size[::-1])  # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])
        # image_0 = np.array(img0)
        # image_1 = np.array(img1)
        # plt.figure()
        # plt.imshow(image_0)
        # plt.show()
        # img0.save('./0.jpg')
        # img1.save('./1.jpg')
        if self._transforms is not None:
            img0, img1, target = self._transforms(img0, img1, target)
        # to_pil = ToPILImage()
        # image0 = to_pil(img0)
        # image1 = to_pil(img1)
        # image0.save('./0_t.jpg')
        # image1.save('./1_t.jpg')
        return img0, img1, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s
```