import math
import torch
from task_configs import get_config
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

        # Creating xgb model
        self.model = get_xgb_model(self.task_type, self.output_dim)
        
        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.total_test_time = 0

#         # no of examples at each step/batch
        self.train_batch_size = 64
        self.test_batch_size = 64
        assert self.task == "spherical"
        if (self.task == "spherical"):
            self.task = "SPHERICAL"
        dims, sample_shape, num_classes, batch_size, epochs, loss, lr, arch_lr, weight_decay, opt, arch_opt, weight_sched_search, weight_sched_train, accum, clip, retrain_clip, validation_freq, retrain_freq, \
        einsum, retrain_epochs, arch_default, kernel_choices_default, dilation_choices_default, quick_search, quick_retrain, config_kwargs = get_config(self.task)  

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

        # If PyTorch dataloader for training set doen't already exists, get the train dataloader
        if not hasattr(self, "trainloader"):
            self.trainloader = self.get_dataloader(
                dataset,
                self.train_batch_size,
                "train",
            )

        train_start = time.time()

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
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            **fit_params,
        )
                
        train_end = time.time()


        train_duration = train_end - train_start
        self.total_train_time += train_duration
        logger.info(
            "{:.2f} sec used for xgboost. ".format(
                train_duration
            )
            + "Total time used for training: {:.2f} sec. ".format(
                self.total_train_time
            )
        )


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
