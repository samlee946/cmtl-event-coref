from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

coref_op_library = tf.load_op_library("./coref_kernels.so")

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")

extract_arguments = coref_op_library.extract_arguments
tf.NotDifferentiable("ExtractArguments")

extract_argument_compatible_scores = coref_op_library.extract_argument_compatible_scores
tf.NotDifferentiable("ExtractArgumentCompatibleScores")

extract_important_argument_incompatible_scores = coref_op_library.extract_important_argument_incompatible_scores
tf.NotDifferentiable("ExtractImportantArgumentIncompatibleScores")

get_predicted_cluster = coref_op_library.get_predicted_cluster
tf.NotDifferentiable("GetPredictedCluster")

extract_important_argument_pairs_violation_scores = coref_op_library.extract_important_argument_pairs_violation_scores
tf.NotDifferentiable("ExtractImportantArgumentPairsViolationScores")

get_entity_sailence_feature = coref_op_library.get_entity_sailence_feature
tf.NotDifferentiable("GetEntitySailenceFeature")

dependency_pruning = coref_op_library.dependency_pruning
tf.NotDifferentiable("DependencyPruning")

dependency_feature = coref_op_library.dependency_feature
tf.NotDifferentiable("DependencyFeature")

extract_argument_feature = coref_op_library.extract_argument_feature
tf.NotDifferentiable("ExtractArgumentFeature")