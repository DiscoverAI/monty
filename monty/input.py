import monty.data as data
from monty.ops import *


class CorruptedInputFunction:
    def __init__(self,
                 dataset_path=FLAGS.dataset_path,
                 num_features=FLAGS.num_features,
                 num_epochs=FLAGS.num_epochs,
                 minimum_expressed_genes=FLAGS.minimum_expressed_genes,
                 minimum_library_size=FLAGS.minimum_library_size,
                 batch_size=FLAGS.batch_size,
                 mask_percentage=FLAGS.mask_percentage,
                 shuffle=True):
        self.dataset_path = dataset_path
        self.num_features = num_features
        self.num_epochs = num_epochs
        self.minimum_expressed_genes = minimum_expressed_genes
        self.minimum_library_size = minimum_library_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mask_percentage = mask_percentage

    def __call__(self, params):
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

        corrupted = self._corrupt_input_data(input_data)

        return corrupted, input_data

    def _corrupt_input_data(self, input_data):
        rand_indexes_of_nonzero = self._get_random_percentage_of_nonzero_indices(input_data)
        return set_index_list_to_zero(rand_indexes_of_nonzero, input_data)

    def _get_random_percentage_of_nonzero_indices(self, input_data):
        randomized_nonzero_indices = tf.random_shuffle(
            tf.where(tf.not_equal(input_data, tf.constant(0, dtype=tf.float32))))
        first_dimension_size = get_size_along_first_dim(randomized_nonzero_indices)
        return tf.slice(randomized_nonzero_indices, [0, 0],
                        [int((self.mask_percentage / 100) * first_dimension_size), 2])
