import math
import os
import torch
import logger
import json
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
from types import SimpleNamespace
import typ
from relax.nas import MixedOptimizer
from dash import MixtureSupernet
from task_configs import get_data, get_config, get_model, get_metric, get_hp_configs, get_optimizer
from task_utils import count_params, print_grad, calculate_stats
from task_configs import get_config
from timeit import default_timer

def train_one_epoch(model, optimizer, scheduler, device, loader, loss, clip, accum, temp, decoder=None, transform=None, lr_sched_iter=False, min_lr=5e-6, scale_grad=False):
    model.train()
                    
    train_loss = 0
    optimizer.zero_grad()

    for i, data in enumerate(loader):
        if transform is not None:
            x, y, z = data
            z = z.to(device)
        else:
            x, y = data 
        
        x, y = x.to(device), y.to(device)
            
        out = model(x)

        if decoder is not None:
            out = decoder.decode(out).view(x.shape[0], -1)
            y = decoder.decode(y).view(x.shape[0], -1)

        if transform is not None:
            out = transform(out, z)
            y = transform(y, z)
                        
        l = loss(out, y)
        l.backward()

        if scale_grad:
            model.scale_grad()

        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        if (i + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += l.item()

        if lr_sched_iter and optimizer.param_groups[0]['lr'] > min_lr:
            scheduler.step()

        if i >= temp - 1:
            break

    if (not lr_sched_iter) and optimizer.param_groups[0]['lr'] > min_lr:
        scheduler.step()

    return train_loss / temp


def evaluate(model, device, loader, loss, metric, n_eval, decoder=None, transform=None, fsd_epoch=None):
    model.eval()
    
    eval_loss, eval_score = 0, 0
    
    if fsd_epoch is None:
        with torch.no_grad():
            for data in loader:
                if transform is not None:
                    x, y, z = data
                    z = z.to(device)
                else:
                    x, y = data
                                    
                x, y = x.to(device), y.to(device)
                out = model(x)
                
                if decoder is not None:
                    out = decoder.decode(out).view(x.shape[0], -1)
                    y = decoder.decode(y).view(x.shape[0], -1)
                                    
                if transform is not None:
                    out = transform(out, z)
                    y = transform(y, z)

                eval_loss += loss(out, y).item()
                eval_score += metric(out, y).item()

        eval_loss /= n_eval
        eval_score /= n_eval

    else:
        outs, ys = [], []
        with torch.no_grad():
            for ix in range(loader.len):

                if fsd_epoch < 100:
                    if ix > 2000: break

                x, y = loader[ix]
                x, y = x.to(device), y.to(device)
                out = model(x).mean(0).unsqueeze(0)
                eval_loss += loss(out, y).item()
                outs.append(torch.sigmoid(out).detach().cpu().numpy()[0])
                ys.append(y.detach().cpu().numpy()[0])

        outs = np.asarray(outs).astype('float32')
        ys = np.asarray(ys).astype('int32')
        stats = calculate_stats(outs, ys)
        eval_score = np.mean([stat['AP'] for stat in stats])
        eval_loss /= n_eval

    return eval_loss, eval_score

class OurModel:
    def __init__(self, metadata):
        '''
        The initalization procedure for your method given the metadata of the task
        '''
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        # Attribute necessary for ingestion program to stop evaluation process
        self.done_training = False
        self.metadata_ = metadata
        self.task = self.metadata_.get_dataset_name()
        self.task_type = self.metadata_.get_task_type()
        
        # Getting details of the data from meta data
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_dim = math.prod(self.metadata_.get_output_shape())

        self.num_examples_train = self.metadata_.size()

        row_count, col_count = self.metadata_.get_tensor_shape()[2:4]
        channel = self.metadata_.get_tensor_shape()[1]
        sequence_size = self.metadata_.get_tensor_shape()[0]

        self.num_train = self.metadata_.size()
        self.num_test = self.metadata_.get_output_shape()

        # Getting the device available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Device Found = ", self.device, "\nMoving Model and Data into the device..."
        )
        assert torch.cuda.is_available() # force xgboost on gpu
        self.input_shape = (channel, sequence_size, row_count, col_count)
        print("\n\nINPUT SHAPE = ", self.input_shape)
        
        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.total_test_time = 0

#         # no of examples at each step/batch
        self.train_batch_size = 64
        self.test_batch_size = 64
    		# ABOVE: HACKATHON
    
        # BELOW: INTEGRATION
        assert self.task == "spherical"
        if (self.task == "spherical"):
            self.task = "SPHERICAL"
        args = SimpleNamespace(
            dataset=self.task,
            arch=None,
            experiment_id=0,
            seed=0,
            kernel_choices=[3,5,7], # IDK
            dilation_choices = [1], # IDK
            verbose=0,
            print_freq = 10,
            valid_split=0,
            reproducibility=0,
            separable=1,
            stream=1,
        )
        self.args = args

        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        exp_id = 'baseline' if args.baseline else args.experiment_id
        args.save_dir = 'results_acc/'  + args.dataset + '/' + ('default' if len(args.arch) == 0 else args.arch) +'/' + exp_id + "/" + str(args.seed)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        with open(args.save_dir + '/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        print("------- Experiment Summary --------")
        print(args.__dict__)

        torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed) 
        torch.cuda.manual_seed_all(args.seed)

        if args.reproducibility:
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            cudnn.benchmark = True
        dims, sample_shape, num_classes, batch_size, epochs, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq, \
        einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs = get_config(self.task)  

        arch = args.arch if len(args.arch) > 0 else arch_default

        if config_kwargs['arch_retrain_default'] is not None:
            arch_retrain = config_kwargs['arch_retrain_default']
        else:
            arch_retrain = arch

        if args.baseline:
            arch = arch if len(args.arch) > 0 else arch_retrain

        kernel_choices = args.kernel_choices if args.kernel_choices[0] is not None else kernel_choices_default
        dilation_choices = args.dilation_choices if args.dilation_choices[0] is not None else dilation_choices_default

        train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs = get_data(args.dataset, batch_size, arch, args.valid_split)
        model = get_model(arch, sample_shape, num_classes, config_kwargs)
        metric, compare_metrics = get_metric(args.dataset)

        train_score, train_time, retrain_score, retrain_time, param_values_list, prev_param_values = [], [], [], [], [], None

        model = MixtureSupernet.create(model.cpu(), in_place=True)

        if not args.baseline:
            model.conv2mixture(torch.zeros(sample_shape),  kernel_sizes=kernel_choices, dilations=dilation_choices, dims=dims, separable=args.separable, 
                stream=args.stream, device=args.device, einsum=einsum, **config_kwargs)

            if dims == 1:
                model.remove_module("chomp")
            opts = [opt(model.model_weights(), lr=lr, weight_decay=weight_decay), arch_opt(model.arch_params(), lr=arch_lr, weight_decay=weight_decay)]

        else:
            opts = [opt(model.model_weights(), lr=lr, weight_decay=weight_decay)]
            epochs = retrain_epochs
            weight_sched_search = weight_sched_train
            clip = retrain_clip

        optimizer = MixedOptimizer(opts)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=weight_sched_search)
        lr_sched_iter = arch == 'convnext'

        decoder = data_kwargs['decoder'] if data_kwargs is not None and 'decoder' in data_kwargs else None 
        transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None 
        n_train_temp = int(quick_search * n_train) + 1 if quick_search < 1 else n_train

        if args.device == 'cuda':
            model.cuda()
            try:
                loss.cuda()
            except:
                pass
            if decoder is not None:
                decoder.cuda()
        self.model = model
        self.decoder= decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr_sched_iter = lr_sched_iter
        self.transform = transform
        self.n_train_temp = n_train_temp
        print("search arch:", arch, "\tretrain arch:", arch_retrain)
        print("batch size:", batch_size, "\tlr:", lr, "\tarch lr:", arch_lr)
        print("arch configs:", config_kwargs)
        print("kernel choices:", kernel_choices_default, "\tdilation choices:", dilation_choices_default)
        print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)

    def get_dataloader(self, dataset, batch_size, split):
        """Get the PyTorch dataloader. Do not modify this method.
        Args:
          dataset:
          batch_size : batch_size for training set
        Return:
          dataloader: PyTorch Dataloader
        """
        if split == "train":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=dataset.collate_fn,
            )
        elif split == "test":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )
        return dataloader

    def train(self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None):
        '''
        The training procedure of your method given training data, validation data (which is only directly provided in certain tasks, otherwise you are free to create your own validation strategies), and remaining time budget for training.
        '''
        
        """Train this algorithm on the Pytorch dataset.
        ****************************************************************************
        ****************************************************************************
        Args:
          dataset: a `DecathlonDataset` object. Each of its examples is of the form
                (example, labels)
              where `example` is a dense 4-D Tensor of shape
                (sequence_size, row_count, col_count, num_channels)
              and `labels` is a 1-D or 2-D Tensor
          val_dataset: a 'DecathlonDataset' object. Is not 'None' if a pre-split validation set is provided, in which case you should use it for any validation purposes. Otherwise, you are free to create your own validation split(s) as desired.
          
          val_metadata: a 'DecathlonMetadata' object, corresponding to 'val_dataset'.
          remaining_time_budget: time remaining to execute train(). The method
              should keep track of its execution time to avoid exceeding its time
              budget. If remaining_time_budget is None, no time budget is imposed.
              
          remaining_time_budget: the time budget constraint for the task, which may influence the training procedure.
        """

        dims, sample_shape, num_classes, batch_size, epochs, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq, \
        einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs = get_config(self.task)  
        args = self.args
        # If PyTorch dataloader for training set doen't already exists, get the train dataloader
        if not hasattr(self, "trainloader"):
            self.trainloader = self.get_dataloader(
                dataset,
                self.train_batch_size,
                "train",
            )

        time_start = default_timer()
        # Training (no loop)
        x_train, y_train = merge_batches(self.trainloader, (self.task_type=="single-label") )
        print(x_train.shape, y_train.shape)
        if val_dataset:
            valloader = self.get_dataloader(val_dataset, self.test_batch_size, "test")
            x_valid, y_valid = merge_batches(valloader, (self.task_type=="single-label") )
        else:
            random_state=None # can set this for reproducibility if desired
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=random_state)
        
        fit_params = {"verbose":True}
        # ABOVE: HACKATHON       
        print("\n------- Start Arch Search --------")
        print("param count:", count_params(self.model))
        for ep in range(epochs):
            train_loss = train_one_epoch(self.model, self.optimizer, self.scheduler, self.args.device, train_loader, loss, clip, 1, self.n_train_temp, self.decoder,
                                         self.transform, self.lr_sched_iter, scale_grad=not args.baseline)

            if args.verbose and not args.baseline and (ep + 1) % args.print_freq == 0:
                print_grad(model, kernel_choices, dilation_choices)

            if ep % validation_freq == 0 or ep == epochs - 1: 
                if args.baseline:
                    val_loss, val_score = evaluate(model, args.device, val_loader, loss, metric, n_val, decoder, transform, fsd_epoch=ep if args.dataset == 'FSD' else None)
                    train_score.append(val_score)

                    if (ep + 1) % args.print_freq == 0  or ep == epochs - 1: 
                        print("[train", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (default_timer() - time_start), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))

                    np.save(os.path.join(args.save_dir, 'train_score.npy'), train_score)

                elif (ep + 1) % args.print_freq == 0 or ep == epochs - 1:
                    print("[train", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (default_timer() - time_start), "\ttrain loss:", "%.4f" % train_loss)

                time_end = default_timer()
                train_time.append(time_end - time_start)
                np.save(os.path.join(args.save_dir, 'train_time.npy'), train_time)

            if not args.baseline and ((ep + 1) % retrain_freq == 0 or ep == epochs - 1):

                param_values, ks, ds = [], [], []
                for name, param in model.named_arch_params():
                    param_values.append(param.data.argmax(0))
                    if args.verbose:
                        print(name, param.data)

                    ks.append(kernel_choices[int(param_values[-1] // len(dilation_choices))])
                    ds.append(dilation_choices[int(param_values[-1] % len(dilation_choices))])

                param_values = torch.stack(param_values, dim = 0)

                print("[searched kernel pattern] ks:", ks, "\tds:", ds)

                if prev_param_values is not None and torch.equal(param_values, prev_param_values):
                    print("\n------- Arch Search Converge --------")
                else:
                    print("\n------- Start Hyperparameter Search --------")

                    loaded = False

                    if os.path.isfile(os.path.join(args.save_dir, 'network_hps.npy')):
                        lr, drop_rate, weight_decay, momentum = np.load(os.path.join(args.save_dir, 'network_hps.npy'))
                        loaded = True
                        print("[load hp] bs = ", batch_size, " lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)

                    else:
                        search_scores = []
                        search_train_loader, search_val_loader, search_test_loader, search_n_train, search_n_val, search_n_test, search_data_kwargs = get_data(args.dataset, accum * batch_size, arch_retrain, True)
                        retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds)
                        retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)

                        retrain_model = retrain_model.to(args.device)
                        torch.save(retrain_model.state_dict(), os.path.join(args.save_dir, 'init.pt'))

                        hp_configs, search_epochs, subsampling_ratio = get_hp_configs(args.dataset, n_train)

                        search_n_temp = int(subsampling_ratio * search_n_train) + 1

                        prev_lr = hp_configs[0][0]
                        best_score_prev = None

                        for lr, drop_rate, weight_decay, momentum in hp_configs:
                            if lr != prev_lr:
                                best_score = compare_metrics(search_scores)

                                if best_score_prev is not None:
                                    if best_score == best_score_prev:
                                        break

                                best_score_prev = best_score
                                prev_lr = lr

                            retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds, dropout = drop_rate)
                            retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)
                            retrain_model = retrain_model.to(args.device)

                            retrain_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'init.pt')))
                            retrain_model.set_arch_requires_grad(False)

                            retrain_optimizer = get_optimizer(momentum=momentum, weight_decay=weight_decay)(retrain_model.parameters(), lr=lr)
                            retrain_scheduler = torch.optim.lr_scheduler.LambdaLR(retrain_optimizer, lr_lambda=weight_sched_train)

                            retrain_time_start = default_timer()

                            for retrain_ep in range(search_epochs):

                                retrain_loss = train_one_epoch(retrain_model, retrain_optimizer, retrain_scheduler, args.device, search_train_loader, loss, retrain_clip, 1, search_n_temp, decoder, transform, lr_sched_iter)

                            retrain_val_loss, retrain_val_score = evaluate(retrain_model, args.device, search_val_loader, loss, metric, search_n_val, decoder, transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
                            retrain_time_end = default_timer()
                            search_scores.append(retrain_val_score)
                            train_time.append(retrain_time_end - retrain_time_start)
                            print("[hp search] bs = ", batch_size, " lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum, " time elapsed:", "%.4f" % (retrain_time_end - retrain_time_start), "\ttrain loss:", "%.4f" % retrain_loss, "\tval loss:", "%.4f" % retrain_val_loss, "\tval score:", "%.4f" % retrain_val_score)
                            del retrain_model

                        idx = np.argwhere(search_scores == compare_metrics(search_scores))[0][0]
                        lr, drop_rate, weight_decay, momentum = hp_configs[idx]
                        np.save(os.path.join(args.save_dir, 'network_hps.npy'), hp_configs[idx])
                        np.save(os.path.join(args.save_dir, 'train_time.npy'), train_time)
                        print("[selected hp] lr = ", "%.6f" % lr, " drop rate = ", "%.2f" % drop_rate, " weight decay = ", "%.6f" % weight_decay, " momentum = ", "%.2f" % momentum)
                        del search_train_loader, search_val_loader

                    print("\n------- Start Retrain --------")
                    retrain_model = get_model(arch_retrain, sample_shape, num_classes, config_kwargs, ks = ks, ds = ds, dropout = drop_rate)
                    retrain_train_loader, retrain_val_loader, retrain_test_loader, retrain_n_train, retrain_n_val, retrain_n_test, data_kwargs = get_data(args.dataset, accum * batch_size, arch_retrain, args.valid_split)
                    retrain_n_temp = int(quick_retrain * retrain_n_train) + 1 if quick_retrain < 1 else retrain_n_train

                    retrain_model = MixtureSupernet.create(retrain_model.cpu(), in_place=True)
                    retrain_model = retrain_model.to(args.device)

                    if not loaded:
                        retrain_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'init.pt')))
                    retrain_optimizer = get_optimizer(momentum = momentum, weight_decay = weight_decay)(retrain_model.parameters(), lr = lr)
                    retrain_model.set_arch_requires_grad(False)

                    retrain_scheduler = torch.optim.lr_scheduler.LambdaLR(retrain_optimizer, lr_lambda = weight_sched_train)

                    time_retrain = 0
                    score = []
                    print("param count:", count_params(retrain_model))
                    for retrain_ep in range(retrain_epochs):
                        retrain_time_start = default_timer()
                        retrain_loss = train_one_epoch(retrain_model, retrain_optimizer, retrain_scheduler, args.device, retrain_train_loader, loss, retrain_clip, 1, retrain_n_temp, decoder, transform, lr_sched_iter)

                        if retrain_ep % validation_freq == 0 or retrain_ep == retrain_epochs - 1:
                            retrain_val_loss, retrain_val_score = evaluate(retrain_model, args.device, retrain_val_loader, loss, metric, retrain_n_val, decoder, transform, fsd_epoch=retrain_ep if args.dataset == 'FSD' else None)

                            retrain_time_end = default_timer()
                            time_retrain += retrain_time_end - retrain_time_start
                            score.append(retrain_val_score)

                            if compare_metrics(score) == retrain_val_score:
                                try:
                                    retrain_model.save_arch(os.path.join(args.save_dir, 'arch.th'))
                                    torch.save(retrain_model.state_dict(), os.path.join(args.save_dir, 'network_weights.pt'))
                                    np.save(os.path.join(args.save_dir, 'retrain_score.npy'), retrain_score)
                                    np.save(os.path.join(args.save_dir, 'retrain_time.npy'), retrain_time)
                                except AttributeError:
                                    pass

                            if (retrain_ep + 1) % args.print_freq == 0  or retrain_ep == retrain_epochs - 1: 
                                print("[retrain", retrain_ep, "%.6f" % retrain_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (retrain_time_end - retrain_time_start), "\ttrain loss:", "%.4f" % retrain_loss, "\tval loss:", "%.4f" % retrain_val_loss, "\tval score:", "%.4f" % retrain_val_score, "\tbest val score:", "%.4f" % compare_metrics(score))

                            if retrain_ep == retrain_epochs - 1:
                                retrain_score.append(score)
                                param_values_list.append(param_values.cpu().detach().numpy())
                                retrain_time.append(time_retrain)
                                prev_param_values = param_values

                                np.save(os.path.join(args.save_dir, 'retrain_score.npy'), retrain_score)
                                np.save(os.path.join(args.save_dir, 'retrain_time.npy'), retrain_time)
                                np.save(os.path.join(args.save_dir, 'network_arch_params.npy'), param_values_list)

                print("\n------- Start Test --------")
                test_scores = []
                test_model = retrain_model
                test_time_start = default_timer()
                test_loss, test_score = evaluate(test_model, args.device, self.retrain_test_loader, loss, self.metric, self.retrain_n_test, self.decoder, self.transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
                test_time_end = default_timer()
                test_scores.append(test_score)

                print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)

                test_model.load_state_dict(torch.load(os.path.join(args.save_dir, 'network_weights.pt')))
                test_time_start = default_timer()
                test_loss, test_score = evaluate(test_model, args.device, self.retrain_test_loader, loss, self.metric, self.retrain_n_test, self.decoder, self.transform, fsd_epoch=200 if args.dataset == 'FSD' else None)
                test_time_end = default_timer()
                test_scores.append(test_score)

                print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
                np.save(os.path.join(args.save_dir, 'test_score.npy'), test_scores)

        # BELOW: HACKTHON
        train_end = time.time()

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the Pytorch dataloader.
        Args:
          Same as that of `train` method, except that the `labels` will be empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
          remaining_time_budget: the remaining time budget left for testing, post-training 
        """

        test_begin = time.time()

        logger.info("Begin testing...")

        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )
        
        x_test, _ = merge_batches(self.testloader, (self.task_type=="single-label") )
        
        # get test predictions from the model
        predictions = self.model.predict(x_test)
        # If the task is multi-class single label, the output will be in raw labels; we need to convert to ohe for passing back to ingestion
        if (self.task_type=="single-label"):
            n = self.metadata_.get_output_shape()[0]
            predictions = np.eye(n)[predictions.astype(int)]
        
        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration

        logger.info(
            "[+] Successfully made one prediction. {:.2f} sec used. ".format(
                test_duration
            )
            + "Total time used for testing: {:.2f} sec. ".format(self.total_test_time)
        )
        return predictions
