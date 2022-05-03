from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import AngularPenaltySMLoss
from torchreid.losses import TripletLoss, CrossEntropyLoss

from ..engine import Engine


class ImageArcFaceEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageArcFaceEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        loss_type='arcface',
        weight_t=0,
        distance='cosine',
        margin=0.3
    ):
        super(ImageArcFaceEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)
        self.criterion_t = TripletLoss(margin=margin, distance=distance)
        self.weight_t = weight_t
        if weight_t > 0:
            self.scheduler = 'RandomIdentitySampler'
        if hasattr(self.model, 'module'):
            in_features = self.model.module.fc.out_features
        else:
            in_features = self.model.fc.out_features
        self.criterion = AngularPenaltySMLoss(
            in_features=in_features,
            out_features=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            loss_type=loss_type
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs = self.model(imgs)

        loss_summary = {}

        loss = self.compute_loss(self.criterion, outputs, pids)

        # add triplet
        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, outputs, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()
        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary['loss_c'] = loss.item()

        return loss_summary
