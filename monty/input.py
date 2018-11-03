import monty.data as data
from monty import FLAGS


class UnchangedInputFunction:
    def __init__(self,
                 mode,
                 dataset_path=FLAGS.dataset_path,
                 num_features=FLAGS.num_features,
                 num_epochs=FLAGS.num_epochs,
                 minimum_expressed_genes=FLAGS.minimum_expressed_genes,
                 minimum_library_size=FLAGS.minimum_library_size,
                 batch_size=FLAGS.batch_size,
                 shuffle=True):
        self.mode = mode
        self.dataset_path = dataset_path
        self.num_features = num_features
        self.num_epochs = num_epochs
        self.minimum_expressed_genes = minimum_expressed_genes
        self.minimum_library_size = minimum_library_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self):
        dataset_file_path = data.download_if_not_present(self.dataset_path)
        dataset = data.create_dataset(dataset_file_path,
                                      num_features=self.num_features,
                                      num_epochs=self.num_epochs,
                                      shuffle=self.shuffle,
                                      shuffle_buffer_size=1000)
        dataset = data.drop_outliers(dataset, self.minimum_library_size, self.minimum_library_size)
        dataset = data.normalize_dataset(dataset)
        iterator = dataset.batch(batch_size=self.batch_size, drop_remainder=True).make_one_shot_iterator()

        input_data = iterator.get_next()

        return input_data, input_data
