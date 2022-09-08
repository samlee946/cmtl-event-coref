#include <map>
#include <limits>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>       /* floor */


using namespace tensorflow;

REGISTER_OP("ExtractSpans")
.Input("span_scores: float32")
.Input("candidate_starts: int32")
.Input("candidate_ends: int32")
.Input("num_output_spans: int32")
.Input("max_sentence_length: int32")
.Attr("sort_spans: bool")
.Output("output_span_indices: int32");

class ExtractSpansOp : public OpKernel {
public:
  explicit ExtractSpansOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sort_spans", &_sort_spans));
  }

  void Compute(OpKernelContext* context) override {
    TTypes<float>::ConstMatrix span_scores = context->input(0).matrix<float>();
    TTypes<int32>::ConstMatrix candidate_starts = context->input(1).matrix<int32>();
    TTypes<int32>::ConstMatrix candidate_ends = context->input(2).matrix<int32>();
    TTypes<int32>::ConstVec num_output_spans = context->input(3).vec<int32>();
    int max_sentence_length = context->input(4).scalar<int32>()();

    int num_sentences = span_scores.dimension(0);
    int num_input_spans = span_scores.dimension(1);
    int max_num_output_spans = 0;
    for (int i = 0; i < num_sentences; i++) {
      if (num_output_spans(i) > max_num_output_spans) {
        max_num_output_spans = num_output_spans(i);
      }
    }

    Tensor* output_span_indices_tensor = nullptr;
    TensorShape output_span_indices_shape({num_sentences, max_num_output_spans});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_span_indices_shape,
                                                     &output_span_indices_tensor));
    TTypes<int32>::Matrix output_span_indices = output_span_indices_tensor->matrix<int32>();

    std::vector<std::vector<int>> sorted_input_span_indices(num_sentences,
                                                            std::vector<int>(num_input_spans));
    for (int i = 0; i < num_sentences; i++) {
      std::iota(sorted_input_span_indices[i].begin(), sorted_input_span_indices[i].end(), 0);
      std::sort(sorted_input_span_indices[i].begin(), sorted_input_span_indices[i].end(),
                [&span_scores, &i](int j1, int j2) {
                  return span_scores(i, j2) < span_scores(i, j1);
                });
    }

    for (int l = 0; l < num_sentences; l++) {
      std::vector<int> top_span_indices;
      std::unordered_map<int, int> end_to_earliest_start;
      std::unordered_map<int, int> start_to_latest_end;

      int current_span_index = 0,
          num_selected_spans = 0;
      while (num_selected_spans < num_output_spans(l) && current_span_index < num_input_spans) {
        int i = sorted_input_span_indices[l][current_span_index];
        bool any_crossing = false;
        const int start = candidate_starts(l, i);
        const int end = candidate_ends(l, i);
        for (int j = start; j <= end; ++j) {
          auto latest_end_iter = start_to_latest_end.find(j);
          if (latest_end_iter != start_to_latest_end.end() && j > start && latest_end_iter->second > end) {
            // Given (), exists [], such that ( [ ) ]
            any_crossing = true;
            break;
          }
          auto earliest_start_iter = end_to_earliest_start.find(j);
          if (earliest_start_iter != end_to_earliest_start.end() && j < end && earliest_start_iter->second < start) {
            // Given (), exists [], such that [ ( ] )
            any_crossing = true;
            break;
          }
        }
        if (!any_crossing) {
          if (_sort_spans) {
            top_span_indices.push_back(i);
          } else {
            output_span_indices(l, num_selected_spans) = i;
          }
          ++num_selected_spans;
          // Update data struct.
          auto latest_end_iter = start_to_latest_end.find(start);
          if (latest_end_iter == start_to_latest_end.end() || end > latest_end_iter->second) {
            start_to_latest_end[start] = end;
          }
          auto earliest_start_iter = end_to_earliest_start.find(end);
          if (earliest_start_iter == end_to_earliest_start.end() || start < earliest_start_iter->second) {
            end_to_earliest_start[end] = start;
          }
        }
        ++current_span_index;
      }
      // Sort and populate selected span indices.
      if (_sort_spans) {
        std::sort(top_span_indices.begin(), top_span_indices.end(),
                  [&candidate_starts, &candidate_ends, &l] (int i1, int i2) {
                    if (candidate_starts(l, i1) < candidate_starts(l, i2)) {
                      return true;
                    } else if (candidate_starts(l, i1) > candidate_starts(l, i2)) {
                      return false;
                    } else if (candidate_ends(l, i1) < candidate_ends(l, i2)) {
                      return true;
                    } else if (candidate_ends(l, i1) > candidate_ends(l, i2)) {
                      return false;
                    } else {
                      return i1 < i2;
                    }
                  });
        for (int i = 0; i < num_output_spans(l); ++i) {
          output_span_indices(l, i) = top_span_indices[i];
        }
      }
      // Pad with the first span index.
      for (int i = num_selected_spans; i < max_num_output_spans; ++i) {
        output_span_indices(l, i) = output_span_indices(l, 0);
      }
    }
  }
private:
  bool _sort_spans;
};

REGISTER_KERNEL_BUILDER(Name("ExtractSpans").Device(DEVICE_CPU), ExtractSpansOp);


REGISTER_OP("ExtractArguments")
.Input("span_starts: int32")
.Input("span_ends: int32")
.Input("arg_mention_starts: int32")
.Input("arg_mention_ends: int32")
.Input("arg_starts: int32")
.Input("arg_ends: int32")
.Input("arg_roles: int32")
.Output("span_arguments: int32");

class ExtractArgumentsOp : public OpKernel {
public:
  explicit ExtractArgumentsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec span_starts = context->input(0).vec<int32>();
    TTypes<int32>::ConstVec span_ends = context->input(1).vec<int32>();
    TTypes<int32>::ConstVec arg_mention_starts = context->input(2).vec<int32>();
    TTypes<int32>::ConstVec arg_mention_ends = context->input(3).vec<int32>();
    TTypes<int32>::ConstVec arg_starts = context->input(4).vec<int32>();
    TTypes<int32>::ConstVec arg_ends = context->input(5).vec<int32>();
    TTypes<int32>::ConstVec arg_roles = context->input(6).vec<int32>();

    int num_mentions = span_starts.dimension(0);
    int num_arg_roles = 20;

    Tensor* span_arguments_tensor = nullptr;
    TensorShape span_arguments_shape({num_mentions, num_arg_roles});
    OP_REQUIRES_OK(context, context->allocate_output(0, span_arguments_shape, &span_arguments_tensor));
    TTypes<int32>::Matrix span_arguments = span_arguments_tensor->matrix<int32>();

    std::map<std::pair<int, int>, int> mention_indices;
    for (int i = 0; i < num_mentions; ++i){
      mention_indices[std::pair<int, int>(span_starts(i), span_ends(i))] = i;
    }

    for (int i = 0; i < num_mentions; ++i){
      for (int j = 0; j < num_arg_roles; ++j){
        span_arguments(i, j) = 0;
      }
    }

    int num_gold_args = arg_mention_starts.dimension(0);
    for (int i = 0; i < num_gold_args; ++i){
      auto midx = mention_indices.find(std::pair<int, int>(arg_mention_starts(i), arg_mention_ends(i)));
      auto argidx = mention_indices.find(std::pair<int, int>(arg_starts(i), arg_ends(i)));
      if (midx != mention_indices.end() && argidx != mention_indices.end()){
        span_arguments(midx->second, arg_roles(i)) = argidx->second + 1;
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ExtractArguments").Device(DEVICE_CPU), ExtractArgumentsOp);


REGISTER_OP("ExtractArgumentCompatibleScores")
.Input("arg_compatible_scores: float32")
.Input("arg_roles: int32")
.Input("num_mentions: int32")
.Input("num_arg_per_mention: int32")
.Input("mention_cluster_ids: int32")
.Input("gold_arg_roles: int32")
.Output("mention_arg_compatible_scores: float32")
.Output("mention_arg_compatible_masks: float32")
.Output("mention_arg_compatible_labels: int32");

class ExtractArgumentCompatibleScoresOp : public OpKernel {
public:
  explicit ExtractArgumentCompatibleScoresOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<float>::ConstMatrix arg_compatible_scores = context->input(0).matrix<float>(); // [k*a, k*a]
    TTypes<int32>::ConstVec arg_roles = context->input(1).vec<int32>();
    int num_mentions = context->input(2).scalar<int32>()();
    int num_arg_per_mention = context->input(3).scalar<int32>()();
    TTypes<int32>::ConstVec mention_cluster_ids = context->input(4).vec<int32>();
    TTypes<int32>::ConstVec gold_arg_roles = context->input(5).vec<int32>();

    Tensor* mention_arg_compatible_scores_tensor = nullptr; // [k*k, a*a]
    TensorShape mention_arg_compatible_scores_shape({num_mentions * num_mentions, num_arg_per_mention * num_arg_per_mention});
    OP_REQUIRES_OK(context, context->allocate_output(0, mention_arg_compatible_scores_shape, &mention_arg_compatible_scores_tensor));
    TTypes<float>::Matrix mention_arg_compatible_scores = mention_arg_compatible_scores_tensor->matrix<float>();

    Tensor* mention_arg_compatible_mask_tensor = nullptr; // [k*k, a*a]
    TensorShape mention_arg_compatible_mask_shape({num_mentions * num_mentions, num_arg_per_mention * num_arg_per_mention});
    OP_REQUIRES_OK(context, context->allocate_output(1, mention_arg_compatible_mask_shape, &mention_arg_compatible_mask_tensor));
    TTypes<float>::Matrix mention_arg_compatible_masks = mention_arg_compatible_mask_tensor->matrix<float>();

    Tensor* mention_arg_compatible_labels_tensor = nullptr; // [k*k, a*a]
    TensorShape mention_arg_compatible_labels_shape({num_mentions * num_mentions, num_arg_per_mention * num_arg_per_mention});
    OP_REQUIRES_OK(context, context->allocate_output(2, mention_arg_compatible_labels_shape, &mention_arg_compatible_labels_tensor));
    TTypes<int>::Matrix mention_arg_compatible_labels = mention_arg_compatible_labels_tensor->matrix<int>();

    int total_num_arg = arg_compatible_scores.dimension(0);

    for(int i = 0; i < total_num_arg; ++i){
      for(int j = 0; j < total_num_arg; ++j){
        int k_i = i / num_arg_per_mention;
        int k_j = j / num_arg_per_mention;
        int mention_idx = num_mentions * k_i + k_j;
        int score_idx = num_arg_per_mention * (i % num_arg_per_mention) + (j % num_arg_per_mention);

        int arg_role_i = arg_roles(i);
        int arg_role_j = arg_roles(j);

        mention_arg_compatible_scores(mention_idx, score_idx) = arg_compatible_scores(i, j);

        int k_i_cid = mention_cluster_ids(k_i);
        int k_j_cid = mention_cluster_ids(k_j);
        int gold_arg_role_i = gold_arg_roles(i);
        int gold_arg_role_j = gold_arg_roles(j);

        if(k_i_cid == k_j_cid && gold_arg_role_i == gold_arg_role_j && gold_arg_role_i != 0){
          mention_arg_compatible_labels(mention_idx, score_idx) = 1;
        }else{
          mention_arg_compatible_labels(mention_idx, score_idx) = 0;
        }

        if(arg_role_i == arg_role_j && arg_role_i != 0){
          mention_arg_compatible_masks(mention_idx, score_idx) = 1.0;
        }else{
          //mention_arg_compatible_scores(mention_idx, score_idx) = -std::numeric_limits<float>::infinity();
          mention_arg_compatible_masks(mention_idx, score_idx) = 0.0;
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ExtractArgumentCompatibleScores").Device(DEVICE_CPU), ExtractArgumentCompatibleScoresOp);


REGISTER_OP("ExtractImportantArgumentIncompatibleScores")
.Input("arg_compatible_scores: float32")
.Input("arg_roles: int32")
.Input("num_mentions: int32")
.Input("num_arg_per_mention: int32")
.Output("mention_arg_compatible_scores: float32")
.Output("mention_arg_compatible_masks: float32");

class ExtractImportantArgumentIncompatibleScoresOp : public OpKernel {
public:
  explicit ExtractImportantArgumentIncompatibleScoresOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<float>::ConstMatrix arg_compatible_scores = context->input(0).matrix<float>(); // [k*a, k*a]
    TTypes<int32>::ConstVec arg_roles = context->input(1).vec<int32>();
    int num_mentions = context->input(2).scalar<int32>()();
    int num_arg_per_mention = context->input(3).scalar<int32>()();
    // TTypes<int32>::ConstVec mention_cluster_ids = context->input(4).vec<int32>();
    // TTypes<int32>::ConstVec gold_arg_roles = context->input(5).vec<int32>();

    Tensor* mention_arg_compatible_scores_tensor = nullptr; // [k*k, a*a]
    TensorShape mention_arg_compatible_scores_shape({num_mentions * num_mentions, num_arg_per_mention * num_arg_per_mention});
    OP_REQUIRES_OK(context, context->allocate_output(0, mention_arg_compatible_scores_shape, &mention_arg_compatible_scores_tensor));
    TTypes<float>::Matrix mention_arg_compatible_scores = mention_arg_compatible_scores_tensor->matrix<float>();

    Tensor* mention_arg_compatible_mask_tensor = nullptr; // [k*k, a*a]
    TensorShape mention_arg_compatible_mask_shape({num_mentions * num_mentions, num_arg_per_mention * num_arg_per_mention});
    OP_REQUIRES_OK(context, context->allocate_output(1, mention_arg_compatible_mask_shape, &mention_arg_compatible_mask_tensor));
    TTypes<float>::Matrix mention_arg_compatible_masks = mention_arg_compatible_mask_tensor->matrix<float>();

    // Tensor* mention_arg_compatible_labels_tensor = nullptr; // [k*k, a*a]
    // TensorShape mention_arg_compatible_labels_shape({num_mentions * num_mentions, num_arg_per_mention * num_arg_per_mention});
    // OP_REQUIRES_OK(context, context->allocate_output(2, mention_arg_compatible_labels_shape, &mention_arg_compatible_labels_tensor));
    // TTypes<int>::Matrix mention_arg_compatible_labels = mention_arg_compatible_labels_tensor->matrix<int>();

    int total_num_arg = arg_compatible_scores.dimension(0);

    for(int i = 0; i < total_num_arg; ++i){
      for(int j = 0; j < total_num_arg; ++j){
        int k_i = i / num_arg_per_mention;
        int k_j = j / num_arg_per_mention;
        int mention_idx = num_mentions * k_i + k_j;
        int score_idx = num_arg_per_mention * (i % num_arg_per_mention) + (j % num_arg_per_mention);

        int arg_role_i = arg_roles(i);
        int arg_role_j = arg_roles(j);

        mention_arg_compatible_scores(mention_idx, score_idx) = arg_compatible_scores(i, j);

        float score = mention_arg_compatible_scores(mention_idx, score_idx);
        //RECIPIENT 3, PERSON 8, ATTACKER 10, GIVER 12, TARGET 15, VICTIM 16, ENTITY 19
        if(score < 0.0 && arg_role_i == arg_role_j && (arg_role_i == 3 || arg_role_i == 8 || arg_role_i == 10 || arg_role_i == 12 || arg_role_i == 15 || arg_role_i == 16 || arg_role_i == 19)){
          mention_arg_compatible_masks(mention_idx, score_idx) = 1.0;
        }else{
          //mention_arg_compatible_scores(mention_idx, score_idx) = -std::numeric_limits<float>::infinity();
          mention_arg_compatible_masks(mention_idx, score_idx) = 0.0;
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ExtractImportantArgumentIncompatibleScores").Device(DEVICE_CPU), ExtractImportantArgumentIncompatibleScoresOp);


REGISTER_OP("ExtractImportantArgumentPairsViolationScores")
.Input("arguments: int32")
.Input("arg_roles: int32")
.Input("arg_mask: int32")
.Input("antecedent_scores: float32")
.Input("antecedents: int32")
.Input("antecedent_masks: int32")
.Input("antecedent_pred_scores: float32")
.Input("num_mentions: int32")
.Input("num_arg_per_mention: int32")
//.Output("mention_arg_compatible_scores: float32")
.Output("arg_violation_scores: float32");

class ExtractImportantArgumentPairsViolationScoresOp : public OpKernel {
public:
  explicit ExtractImportantArgumentPairsViolationScoresOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstMatrix arguments = context->input(0).matrix<int32>(); // [k, a]
    TTypes<int32>::ConstMatrix arg_roles = context->input(1).matrix<int32>(); //[k, a]
    TTypes<int32>::ConstMatrix arg_mask = context->input(2).matrix<int32>(); //[k, a]
    TTypes<float>::ConstMatrix antecedent_scores = context->input(3).matrix<float>(); // [k, c]
    TTypes<int32>::ConstMatrix antecedents = context->input(4).matrix<int32>(); // [k, c]
    TTypes<int32>::ConstMatrix antecedent_masks = context->input(5).matrix<int32>(); // [k, c]
    TTypes<float>::ConstVec antecedent_pred_scores = context->input(6).vec<float>(); // [k]

    int num_mentions = context->input(7).scalar<int32>()();
    int num_arg_per_mention = context->input(8).scalar<int32>()();

    int num_antecedents = antecedents.dimension(1); // c

    Tensor* arg_violation_scores_tensor = nullptr; // [k, c]
    TensorShape arg_violation_scores_shape({num_mentions, num_antecedents});

    OP_REQUIRES_OK(context, context->allocate_output(0, arg_violation_scores_shape, &arg_violation_scores_tensor));
    TTypes<float>::Matrix arg_violation_scores = arg_violation_scores_tensor->matrix<float>();

    std::map<std::pair<int, int>, float> antscores_map;
    for(int i = 0; i < num_mentions; i++){
      for(int j = 0; j < num_antecedents; j++){
        int mask = antecedent_masks(i, j);
        float score = antecedent_scores(i, j);

        if(mask == 1){
          antscores_map.insert(std::make_pair(std::make_pair(i, antecedents(i, j)), score));
        }
      }
    }

    for(int i = 0; i < num_mentions; i++){
      for(int j = 0; j < num_antecedents; j++){
        float score = 0.0;

        int mask = antecedent_masks(i, j);
        if(mask == 0){
          arg_violation_scores(i, j) = score;
          continue;
        }

        float argpaircount = 0.0;
        for(int ai = 0; ai < num_arg_per_mention; ai++){
          for(int aj = 0; aj < num_arg_per_mention; aj++){
            // check every argument pair
            int arg_spani = arguments(i, ai);
            int arg_spanj = arguments(j, aj);

            int mask_ai = arg_mask(i, ai);
            int mask_aj = arg_mask(j, aj);

            if (mask_ai != 1 || mask_aj != 1){
              continue;
            }

            int role_ai = arg_roles(i, ai);
            int role_aj = arg_roles(j, aj);

            //RECIPIENT 3, PERSON 8, ATTACKER 10, GIVER 12, TARGET 15, VICTIM 16, ENTITY 19
            if (role_ai == role_aj && (role_ai == 3 || role_ai == 8 || role_ai == 10 || role_ai == 12 || role_ai == 15 || role_ai == 16 )){ //|| role_ai == 19)){
              //
              if(arg_spani > arg_spanj && antscores_map.count(std::make_pair(arg_spani, arg_spanj)) != 0){
                float s =  antecedent_pred_scores(arg_spani) - antscores_map[std::make_pair(arg_spani, arg_spanj)];
                score += s;
                //std::cout << "arg_spani s:" << s << std::endl;
              }

              if(arg_spani < arg_spanj && antscores_map.count(std::make_pair(arg_spanj, arg_spani)) != 0){
                float s = antecedent_pred_scores(arg_spanj) - antscores_map[std::make_pair(arg_spanj, arg_spani)];
                score += s;
                //std::cout << "arg_spanj s:" << s << std::endl;
              }

              argpaircount += 1;
            }

            //std::cout << "m:" << i << ", ant:" << j << ", argm:" << arg_spani << ", argant:" << arg_spanj << ", score:" << score << std::endl;
          }
        }

        if(argpaircount > 0){
          arg_violation_scores(i, j) = score / argpaircount;
        }
        else{
          arg_violation_scores(i, j) = score;
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ExtractImportantArgumentPairsViolationScores").Device(DEVICE_CPU), ExtractImportantArgumentPairsViolationScoresOp);


REGISTER_OP("GetPredictedCluster")
//.Input("span_starts: int32")
//.Input("span_ends: int32")
.Input("antecedents: int32")
.Input("antecedent_scores: float32")
.Input("num_mentions: int32")
.Input("num_antecedents: int32")
.Output("predicted_cluster_indices: int32");

class GetPredictedClusterOp : public OpKernel {
public:
  explicit GetPredictedClusterOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //TTypes<int32>::ConstVec span_starts = context->input(0).vec<int32>();
    //TTypes<int32>::ConstVec span_ends = context->input(1).vec<int32>();
    TTypes<int>::ConstMatrix antecedents = context->input(0).matrix<int32>(); // [k, c]
    TTypes<float>::ConstMatrix antecedent_scores = context->input(1).matrix<float>(); // [k, c+1]
    int num_mentions = context->input(2).scalar<int32>()();
    int num_antecedents = context->input(3).scalar<int32>()();

    Tensor* predicted_cluster_id_tensor = nullptr;
    TensorShape predicted_cluster_id_shape({num_mentions, 1});
    OP_REQUIRES_OK(context, context->allocate_output(0, predicted_cluster_id_shape, &predicted_cluster_id_tensor));
    TTypes<int32>::Matrix predicted_cluster_indices = predicted_cluster_id_tensor->matrix<int32>();

    int k = antecedent_scores.dimension(0);
    int c = antecedent_scores.dimension(1);
    //std::cout << "antscores:" << k << " " << c << std::endl;

    k = antecedents.dimension(0);
    c = antecedents.dimension(1);
    //std::cout << "ants:" << k << " " << c << std::endl;

    std::vector<int> predicted_antecedents;
    for(int i = 0; i < num_mentions; i++){
      int predicted_index = 0;
      float maxscore = 0.0;

      for(int j = 0; j < num_antecedents + 1; j++){
        if(antecedent_scores(i, j) > maxscore){
          maxscore = antecedent_scores(i, j);
          predicted_index = j;
        }
      }
      predicted_index -= 1;

      if(predicted_index < 0){
        predicted_antecedents.push_back(-1);
        //std::cout << i << " ant idx:" << predicted_index << " ant: -1" << std::endl;

      }else{
        predicted_antecedents.push_back(antecedents(i, predicted_index));
        //std::cout << i << " ant idx:" << predicted_index << std::endl;
        //std::cout << "ant:" << antecedents(i, predicted_index) << std::endl;

      }
    }

    std::unordered_map<int, int> mention_to_predicted;
    std::vector<std::vector<int>> predicted_clusters;

    for(int i = 0; i<predicted_antecedents.size(); i++){
      int predicted_antecedent = predicted_antecedents[i];

      // mention i starts a new cluster
      if(predicted_antecedent < 0){
        mention_to_predicted[i] = predicted_clusters.size();
        std::vector<int> predicted_cluster = {i};
        predicted_clusters.push_back(predicted_cluster);
        continue;
      }

      auto predicted_ant_iter = mention_to_predicted.find(predicted_antecedent);
      int predicted_cluster_id;
      if(predicted_ant_iter != mention_to_predicted.end()){
        predicted_cluster_id = predicted_ant_iter->second;
      }else{
        predicted_cluster_id = predicted_clusters.size();
        std::vector<int> predicted_cluster = {predicted_antecedent};
        predicted_clusters.push_back(predicted_cluster);
        mention_to_predicted[predicted_antecedent] = predicted_cluster_id;
      }

      predicted_clusters[predicted_cluster_id].push_back(i);
      mention_to_predicted[i] = predicted_cluster_id;
    }


    for(int i = 0; i< num_mentions; i++){
      auto predicted_cid_iter = mention_to_predicted.find(i);
      predicted_cluster_indices(i, 0)= predicted_cid_iter->second;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("GetPredictedCluster").Device(DEVICE_CPU), GetPredictedClusterOp);


REGISTER_OP("GetEntitySailenceFeature")
.Input("antecedents: int32")
.Input("entity_sailence_scores: float32")
.Input("thres: float32")
//.Input("num_antecedents: int32")
.Output("antecedent_sailence_feature: float32");

class GetEntitySailenceFeatureOp : public OpKernel {
public:
  explicit GetEntitySailenceFeatureOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    //TTypes<int32>::ConstVec span_starts = context->input(0).vec<int32>();
    //TTypes<int32>::ConstVec span_ends = context->input(1).vec<int32>();
    TTypes<int>::ConstMatrix antecedents = context->input(0).matrix<int32>(); // [k, c]
    TTypes<float>::ConstMatrix entity_sailence_scores = context->input(1).matrix<float>(); // [k, emb]

    float thres = context->input(2).scalar<float>()();
    //int num_antecedents = context->input(3).scalar<int32>()();

    //std::cout << "sailence score dim:" << entity_sailence_scores.dimension(0) << "," << entity_sailence_scores.dimension(1) << std::endl;

    int k = antecedents.dimension(0);
    int c = antecedents.dimension(1);

    //std::cout << k << ", " << c << std::endl;

    Tensor* antecedent_sailence_feature_tensor = nullptr;
    TensorShape antecedent_sailence_feature_shape({k, c});
    OP_REQUIRES_OK(context, context->allocate_output(0, antecedent_sailence_feature_shape, &antecedent_sailence_feature_tensor));
    TTypes<float>::Matrix antecedent_sailence_feature = antecedent_sailence_feature_tensor->matrix<float>();

    for(int i = 0; i < k; i++){
      for(int j = 0; j < c; j++){
        int ant_idx = antecedents(i, j);

        if(ant_idx == -1){
          //std::cout << "i:" << i << ", j:" << j << ",ant:" << ant_idx << std::endl; //", rfloor:" << rfloor << std::endl; 
          continue;
        }

        float entity_count = 0.0;
        for(int l = 0; l < 200; l++){
          float scorei = entity_sailence_scores(i, l);
          float scorej = entity_sailence_scores(ant_idx, l);

          if(scorei > 0 && scorej > 0){
            if(scorei > thres and scorej > thres){
              entity_count = entity_count + 1.0;
            }
          }
        }

        antecedent_sailence_feature(i, j) = entity_count;
        
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("GetEntitySailenceFeature").Device(DEVICE_CPU), GetEntitySailenceFeatureOp);

// class GetEntitySailenceFeatureOp : public OpKernel {
// public:
//   explicit GetEntitySailenceFeatureOp(OpKernelConstruction* context) : OpKernel(context) {}

//   void Compute(OpKernelContext* context) override {
//     //TTypes<int32>::ConstVec span_starts = context->input(0).vec<int32>();
//     //TTypes<int32>::ConstVec span_ends = context->input(1).vec<int32>();
//     TTypes<int>::ConstMatrix antecedents = context->input(0).matrix<int32>(); // [k, c]
//     TTypes<float>::ConstMatrix entity_sailence_scores = context->input(1).matrix<float>(); // [k, emb]

//     int num_mentions = context->input(2).scalar<int32>()();
//     //int num_antecedents = context->input(3).scalar<int32>()();

//     //std::cout << "sailence score dim:" << entity_sailence_scores.dimension(0) << "," << entity_sailence_scores.dimension(1) << std::endl;

//     int k = antecedents.dimension(0);
//     int c = antecedents.dimension(1);

//     //std::cout << k << ", " << c << std::endl;

//     Tensor* antecedent_sailence_feature_tensor = nullptr;
//     TensorShape antecedent_sailence_feature_shape({k, c*6});
//     OP_REQUIRES_OK(context, context->allocate_output(0, antecedent_sailence_feature_shape, &antecedent_sailence_feature_tensor));
//     TTypes<float>::Matrix antecedent_sailence_feature = antecedent_sailence_feature_tensor->matrix<float>();

//     for(int i = 0; i < k; i++){
//       for(int j = 0; j < c; j++){
//         int ant_idx = antecedents(i, j);
//         std::vector<float> feature = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

//         if(ant_idx == -1){
//           //std::cout << "i:" << i << ", j:" << j << ",ant:" << ant_idx << std::endl; //", rfloor:" << rfloor << std::endl; 
//           continue;
//         }

//         for(int l = 0; l < 200; l++){
//           float scorei = entity_sailence_scores(i, l);
//           float scorej = entity_sailence_scores(ant_idx, l);

//           if(scorei > 0 && scorej > 0){
//             float ratio = scorei/scorej;
//             if(scorei < scorej){
//               ratio = scorej/scorei;
//             }

//             int rfloor = (int) floor(ratio);

//             if(rfloor < 5 && rfloor > 0){
//               feature[rfloor] = feature[rfloor] + 1.0;
//             }else if(rfloor > 5){
//               feature[5] = feature[5] + 1.0; 
//             }
//           }
//         }

//         for(int l = 0; l < 6; l++){
//           //std::cout << "i:" << i << ", j:" << j << ":" << feature[l] << std::endl;
//           antecedent_sailence_feature(i, j*6+l) = feature[l];
//         }
//       }
//     }
//   }
// };
// REGISTER_KERNEL_BUILDER(Name("GetEntitySailenceFeature").Device(DEVICE_CPU), GetEntitySailenceFeatureOp);


REGISTER_OP("DependencyPruning")
.Input("event_span_ends: int32")
.Input("event_top_antecedents: int32")
.Input("event_top_antecedents_masks: float32")
.Input("entity_span_ends: int32")
.Input("entity_cluster_indices: int32")
.Input("dep_child_indices: int32")
.Input("dep_parent_indices: int32")
.Input("dep_paths: int32")
//.Output("top_antecedents: int32")
.Output("top_antecedents_masks: int32");

class DependencyPruningOp : public OpKernel {
public:
  explicit DependencyPruningOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec event_span_ends = context->input(0).vec<int32>();
    TTypes<int>::ConstMatrix event_top_antecedents = context->input(1).matrix<int32>(); // [k, c]
    TTypes<float>::ConstMatrix event_top_antecedents_masks = context->input(2).matrix<float>(); // [k, c]
    TTypes<int32>::ConstVec entity_span_ends = context->input(3).vec<int32>();
    TTypes<int32>::ConstVec entity_cluster_indices = context->input(4).vec<int32>();
    TTypes<int32>::ConstVec dep_child_indices = context->input(5).vec<int32>();
    TTypes<int32>::ConstVec dep_parent_indices = context->input(6).vec<int32>();
    TTypes<int32>::ConstVec dep_paths = context->input(7).vec<int32>();

    int num_mentions = event_top_antecedents.dimension(0);
    int num_antecedents = event_top_antecedents.dimension(1);

    //Tensor* top_antecedents_tensor = nullptr;
    //TensorShape top_antecedents_shape({num_mentions, num_antecedents});
    //OP_REQUIRES_OK(context, context->allocate_output(0, top_antecedents_shape, &top_antecedents_tensor));
    //TTypes<int32>::Matrix top_antecedents = top_antecedents_tensor->matrix<int32>();

    Tensor* top_antecedents_masks_tensor = nullptr;
    TensorShape top_antecedents_masks_shape({num_mentions, num_antecedents});
    OP_REQUIRES_OK(context, context->allocate_output(0, top_antecedents_masks_shape, &top_antecedents_masks_tensor));
    TTypes<int32>::Matrix top_antecedents_masks = top_antecedents_masks_tensor->matrix<int32>();

    std::map<std::pair<int, int>, int> dep_map;
    for (int i = 0; i < dep_parent_indices.size(); ++i){
      dep_map[std::pair<int, int>(dep_child_indices(i), dep_parent_indices(i))] = dep_paths(i);
    }

    std::map<int, int> entity_coref_map;
    for (int i = 0; i < entity_cluster_indices.size(); ++i){
      entity_coref_map[entity_span_ends(i)] = entity_cluster_indices(i);
    }


    for (int i = 0; i < num_mentions; ++i){
      int mention_end = event_span_ends(i);

      // find all entities that have dependency relation with mention
      std::vector<int> entities;
      for(int k = 0; k < num_mentions; ++k){
        auto eidx = dep_map.find(std::pair<int, int>(mention_end, entity_span_ends(k)));
        if (eidx != dep_map.end()){
          entities.push_back(entity_span_ends(k));
        }
      }

      for (int j = 0; j < num_antecedents; ++j){
        //top_antecedents(i, j) = event_top_antecedents(i, j)

        if (event_top_antecedents_masks(i, j) > 0){
          top_antecedents_masks(i, j) = 1;

          int ant_end = event_span_ends(event_top_antecedents(i, j));

          if (mention_end == ant_end){
            continue;
          }

          // find all entities that have dependency relation with antecedent
          std::vector<int> ant_entities;
          for(int k = 0; k < num_mentions; ++k){
            auto eidx = dep_map.find(std::pair<int, int>(ant_end, entity_span_ends(k)));
            if (eidx != dep_map.end()){
              ant_entities.push_back(entity_span_ends(k));
            }
          }

          for(int l = 0; l < entities.size(); ++l){
            auto en_dep_iter = dep_map.find(std::pair<int, int>(mention_end, entities[l]));
            for(int m = 0; m < ant_entities.size(); ++m){
              auto anten_dep_iter = dep_map.find(std::pair<int, int>(ant_end, ant_entities[m]));

              if(en_dep_iter->second == anten_dep_iter->second){
                auto en_cid_iter = entity_coref_map.find(entities[l]);
                auto anten_cid_iter = entity_coref_map.find(ant_entities[m]);
                //std::cout << "mention:" << mention_end << " ant:" << ant_end << "entities:" << entities[l] << ", " << ant_entities[m] << std::endl;
                //std::cout << "mention entity dep:" << en_dep_iter->second << "cid:" << en_cid_iter->second << " ant entity dep:" << anten_dep_iter->second << " cid:" << anten_cid_iter->second << std::endl;
                
                if(en_cid_iter != entity_coref_map.end() && anten_cid_iter != entity_coref_map.end() && en_cid_iter->second != anten_cid_iter->second){
                  //std::cout << "mention entity cid:" << en_cid_iter->second << " ant entity cid:" << anten_cid_iter->second << std::endl;
                  top_antecedents_masks(i, j) = 0;
                  break;
                }
              }
            }
          }
        }else{
          top_antecedents_masks(i, j) = 0;
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("DependencyPruning").Device(DEVICE_CPU), DependencyPruningOp);


REGISTER_OP("DependencyFeature")
.Input("event_span_ends: int32")
.Input("event_top_antecedents: int32")
.Input("event_top_antecedents_masks: float32")
.Input("entity_span_ends: int32")
.Input("entity_cluster_indices: int32")
.Input("dep_child_indices: int32")
.Input("dep_parent_indices: int32")
.Input("dep_paths: int32")
//.Output("top_antecedents: int32")
.Output("top_antecedents_fea: int32");

class DependencyFeatureOp : public OpKernel {
public:
  explicit DependencyFeatureOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec event_span_ends = context->input(0).vec<int32>();
    TTypes<int>::ConstMatrix event_top_antecedents = context->input(1).matrix<int32>(); // [k, c]
    TTypes<float>::ConstMatrix event_top_antecedents_masks = context->input(2).matrix<float>(); // [k, c]
    TTypes<int32>::ConstVec entity_span_ends = context->input(3).vec<int32>();
    TTypes<int32>::ConstVec entity_cluster_indices = context->input(4).vec<int32>();
    TTypes<int32>::ConstVec dep_child_indices = context->input(5).vec<int32>();
    TTypes<int32>::ConstVec dep_parent_indices = context->input(6).vec<int32>();
    TTypes<int32>::ConstVec dep_paths = context->input(7).vec<int32>();

    int num_mentions = event_top_antecedents.dimension(0);
    int num_antecedents = event_top_antecedents.dimension(1);

    //Tensor* top_antecedents_tensor = nullptr;
    //TensorShape top_antecedents_shape({num_mentions, num_antecedents});
    //OP_REQUIRES_OK(context, context->allocate_output(0, top_antecedents_shape, &top_antecedents_tensor));
    //TTypes<int32>::Matrix top_antecedents = top_antecedents_tensor->matrix<int32>();

    // possible values: 0 - no common dep, 1 - has common dep, 2 - has common dep, entity not coref
    Tensor* top_antecedents_fea_tensor = nullptr;
    TensorShape top_antecedents_fea_shape({num_mentions, num_antecedents});
    OP_REQUIRES_OK(context, context->allocate_output(0, top_antecedents_fea_shape, &top_antecedents_fea_tensor));
    TTypes<int32>::Matrix top_antecedents_fea = top_antecedents_fea_tensor->matrix<int32>();

    std::map<std::pair<int, int>, int> dep_map;
    for (int i = 0; i < dep_parent_indices.size(); ++i){
      dep_map[std::pair<int, int>(dep_child_indices(i), dep_parent_indices(i))] = dep_paths(i);
    }

    std::map<int, int> entity_coref_map;
    for (int i = 0; i < entity_cluster_indices.size(); ++i){
      entity_coref_map[entity_span_ends(i)] = entity_cluster_indices(i);
    }


    for (int i = 0; i < num_mentions; ++i){
      int mention_end = event_span_ends(i);

      // find all entities that have dependency relation with mention
      std::vector<int> entities;
      for(int k = 0; k < num_mentions; ++k){
        auto eidx = dep_map.find(std::pair<int, int>(mention_end, entity_span_ends(k)));
        if (eidx != dep_map.end()){
          entities.push_back(entity_span_ends(k));
        }
      }

      for (int j = 0; j < num_antecedents; ++j){
        //top_antecedents(i, j) = event_top_antecedents(i, j)

        if (event_top_antecedents_masks(i, j) > 0){
          top_antecedents_fea(i, j) = 0;

          int ant_end = event_span_ends(event_top_antecedents(i, j));

          if (mention_end == ant_end){
            continue;
          }

          // find all entities that have dependency relation with antecedent
          std::vector<int> ant_entities;
          for(int k = 0; k < num_mentions; ++k){
            auto eidx = dep_map.find(std::pair<int, int>(ant_end, entity_span_ends(k)));
            if (eidx != dep_map.end()){
              ant_entities.push_back(entity_span_ends(k));
            }
          }

          for(int l = 0; l < entities.size(); ++l){
            auto en_dep_iter = dep_map.find(std::pair<int, int>(mention_end, entities[l]));
            for(int m = 0; m < ant_entities.size(); ++m){
              auto anten_dep_iter = dep_map.find(std::pair<int, int>(ant_end, ant_entities[m]));

              if(en_dep_iter->second == anten_dep_iter->second){
                top_antecedents_fea(i, j) = 1;
                auto en_cid_iter = entity_coref_map.find(entities[l]);
                auto anten_cid_iter = entity_coref_map.find(ant_entities[m]);
                //std::cout << "mention:" << mention_end << " ant:" << ant_end << "entities:" << entities[l] << ", " << ant_entities[m] << std::endl;
                //std::cout << "mention entity dep:" << en_dep_iter->second << "cid:" << en_cid_iter->second << " ant entity dep:" << anten_dep_iter->second << " cid:" << anten_cid_iter->second << std::endl;
                
                if(en_cid_iter != entity_coref_map.end() && anten_cid_iter != entity_coref_map.end() && en_cid_iter->second != anten_cid_iter->second){
                  //std::cout << "mention entity cid:" << en_cid_iter->second << " ant entity cid:" << anten_cid_iter->second << std::endl;
                  top_antecedents_fea(i, j) = 2;
                  break;
                }
              }
            }
          }
        }else{
          top_antecedents_fea(i, j) = 0;
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("DependencyFeature").Device(DEVICE_CPU), DependencyFeatureOp);



REGISTER_OP("ExtractArgumentFeature")
.Input("top_span_starts: int32")
.Input("top_span_ends: int32")
.Input("top_antecedents: int32")
.Input("top_antecedents_masks: float32")
.Input("entity_span_starts: int32")
.Input("entity_span_ends: int32")
.Input("entity_cluster_indices: int32")
.Input("arg_mention_starts: int32")
.Input("arg_mention_ends: int32")
.Input("arg_starts: int32")
.Input("arg_ends: int32")
.Input("arg_roles: int32")
.Input("num_arg_roles: int32")
//.Output("top_antecedents: int32")
.Output("top_span_argument_feas: int32");

class ExtractArgumentFeatureOp : public OpKernel {
public:
  explicit ExtractArgumentFeatureOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec top_span_starts = context->input(0).vec<int32>();
    TTypes<int32>::ConstVec top_span_ends = context->input(1).vec<int32>();
    TTypes<int32>::ConstMatrix top_antecedents = context->input(2).matrix<int32>(); // [k, c]
    TTypes<float>::ConstMatrix top_antecedents_masks = context->input(3).matrix<float>(); // [k, c]
    TTypes<int32>::ConstVec entity_span_starts = context->input(4).vec<int32>();
    TTypes<int32>::ConstVec entity_span_ends = context->input(5).vec<int32>();
    TTypes<int32>::ConstVec entity_cluster_indices = context->input(6).vec<int32>();
    TTypes<int32>::ConstVec arg_mention_starts = context->input(7).vec<int32>();
    TTypes<int32>::ConstVec arg_mention_ends = context->input(8).vec<int32>();
    TTypes<int32>::ConstVec arg_starts = context->input(9).vec<int32>();
    TTypes<int32>::ConstVec arg_ends = context->input(10).vec<int32>();
    TTypes<int32>::ConstVec arg_roles = context->input(11).vec<int32>();
    int num_arg_roles = context->input(12).scalar<int32>()();

    int num_mentions = top_antecedents.dimension(0); // k
    int num_antecedents = top_antecedents.dimension(1); // c

    Tensor* top_span_argument_feas_tensor = nullptr;
    TensorShape top_span_argument_feas_shape({num_mentions*num_antecedents, num_arg_roles * 3}); // [k*c, emb]
    OP_REQUIRES_OK(context, context->allocate_output(0, top_span_argument_feas_shape, &top_span_argument_feas_tensor));
    TTypes<int32>::Matrix top_span_argument_feas = top_span_argument_feas_tensor->matrix<int32>();

    std::map<std::pair<int, int>, int> coref_map;
    for (int i = 0; i < entity_cluster_indices.size(); ++i){
      coref_map[std::pair<int, int>(entity_span_starts(i), entity_span_ends(i))] = entity_cluster_indices(i);
    }

    std::map<std::pair<int, int>, std::vector<std::vector<int>>> argument_map;
    for (int i = 0; i < arg_mention_starts.size(); ++i){
      if(arg_roles(i) == 0){ // null arg role
        continue;
      }
      argument_map[std::pair<int, int>(arg_mention_starts(i), arg_mention_ends(i))].push_back({arg_starts(i), arg_ends(i), arg_roles(i) - 1});
    }

    for (int i = 0; i < num_mentions; ++i){
      for (int j = 0; j < num_antecedents; ++j){
        int idx = i * num_antecedents + j;
        for(int k = 0; k < num_arg_roles; k++){
          top_span_argument_feas(idx, k * 3) = 0.0;
          top_span_argument_feas(idx, k * 3 + 1) = 0.0;
          top_span_argument_feas(idx, k * 3 + 2) = 0.0;
        }
      }
    }


    for (int i = 0; i < num_mentions; ++i){
      int mention_start = top_span_starts(i);
      int mention_end = top_span_ends(i);

      auto mention_args_iter = argument_map.find(std::pair<int, int>(mention_start, mention_end));

      for (int j = 0; j < num_antecedents; ++j){
        if (top_antecedents_masks(i, j) < 1){
          int idx = i * num_antecedents + j;
          for(int k = 0; k < num_arg_roles; k++){
            top_span_argument_feas(idx, k * 3) = 0.0;
            top_span_argument_feas(idx, k * 3 + 1) = 0.0;
            top_span_argument_feas(idx, k * 3 + 2) = 0.0;
          }
          continue;
        }

        // event mention doesn't have any arguments
        if( mention_args_iter == argument_map.end()){
          // no common argument
          int idx = i * num_antecedents + j;
          for(int k = 0; k < num_arg_roles; k++){
            top_span_argument_feas(idx, k * 3) = 1.0;
            top_span_argument_feas(idx, k * 3 + 1) = 0.0;
            top_span_argument_feas(idx, k * 3 + 2) = 0.0;
          }

        }else{
          int ant_start = top_span_starts(top_antecedents(i, j));
          int ant_end = top_span_ends(top_antecedents(i, j));

          auto ant_args_iter = argument_map.find(std::pair<int, int>(ant_start, ant_end));

          int idx = i * num_antecedents + j;

          if(ant_args_iter == argument_map.end()){

            // no common argument
            for(int k = 0; k < num_arg_roles; k++){
              top_span_argument_feas(idx, k * 3) = 1.0;
              top_span_argument_feas(idx, k * 3 + 1) = 0.0;
              top_span_argument_feas(idx, k * 3 + 2) = 0.0;
            }
          }else{
            for(int k = 0; k < num_arg_roles; ++k){
              bool find_common_arg = false;
              int feaidx = 0;
              for(int m = 0; m < mention_args_iter->second.size(); m++){
                std::vector<int> marg = mention_args_iter->second[m];

                if(marg[2] == k){
                  for(int n = 0; n < ant_args_iter->second.size(); n++){
                    std::vector<int> narg = ant_args_iter->second[n];

                    if(narg[2] == k){ // common argument role, check if corresponding entity mention are coreferent
                      find_common_arg = true;
                      auto en_cid_iter = coref_map.find(std::pair<int, int>(marg[0], marg[1]));
                      auto anten_cid_iter = coref_map.find(std::pair<int, int>(narg[0], narg[1]));
                      //std::cout << "mention:" << mention_end << " ant:" << ant_end << "entities:" << entities[l] << ", " << ant_entities[m] << std::endl;
                      //std::cout << "mention entity dep:" << en_dep_iter->second << "cid:" << en_cid_iter->second << " ant entity dep:" << anten_dep_iter->second << " cid:" << anten_cid_iter->second << std::endl;
                
                      if(en_cid_iter != coref_map.end() && anten_cid_iter != coref_map.end() && en_cid_iter->second != anten_cid_iter->second){
                        // not coref
                        feaidx = 2;
                      }else if(en_cid_iter != coref_map.end() && anten_cid_iter != coref_map.end() && en_cid_iter->second == anten_cid_iter->second){
                        // coref
                        feaidx = 1;
                      }
                    }
                  }
                }
              }

              // if(feaidx != 0){
              //   std::cout << "arg role:" << k << "feaidx:" << feaidx << std::endl;
              // }

              top_span_argument_feas(idx, k * 3 + feaidx) = 1.0;
              for (int fid = 0; fid < 3; fid++){
                if(fid != feaidx){
                  top_span_argument_feas(idx, k * 3 + fid) = 0.0;
                }
              }

            }

            // std::cout << "k:" << i << " c:" << j << std::endl;
            // int idx = i * num_antecedents + j;
            // for(int fid = 0; fid < num_arg_roles*3; fid++){
            //   std::cout << "(" << fid << "," << top_span_argument_feas(idx, fid) << "), ";
            // }
            // std::cout << "\n" << std::endl;
          }
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ExtractArgumentFeature").Device(DEVICE_CPU), ExtractArgumentFeatureOp);



