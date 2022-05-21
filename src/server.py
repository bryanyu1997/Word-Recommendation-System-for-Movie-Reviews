import gensim.models.keyedvectors as word2vec
from transformers import AutoTokenizer, AutoModel

from src.dataset import get_dataset_and_pos
from src.classifier import load_classifier_model, inference
from config.config import Config
from utils.utils import search_candidate_words, word2vec_array, \
                        eval_distance, hugging_face_distance, \
                        final_wordsandquery_dict, print_log


class Server:
    def __init__(self, cfg=Config):
        self.cfg = Config

        # load dataset
        print_log("Loading gallery dataset")
        self.gallery_set, self.pos_dataset = get_dataset_and_pos(self.cfg)

        # load classifying model
        print_log("Loading classifying model")
        self.model, self.tokenizer = load_classifier_model(self.cfg)

        # load word2vec model
        print_log("Loading word2vec model")
        self.w2v_model = word2vec.KeyedVectors.load_word2vec_format(
                self.cfg.w2v_model_name, binary=True)

        # load huggingface model
        print_log("Loading huggingface model")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.hugging_face_model_name)
        self.hf_model = AutoModel.from_pretrained(
                self.cfg.hugging_face_model_name)

    def inference(self, query_set):
        print_log("Start inferencing")
        final_list = inference(self.model,
                               query_set,
                               self.cfg.BATCH_SIZE,
                               self.tokenizer,
                               self.cfg.MAX_TOKEN_LENGTH,
                               self.cfg.THRESHOLD,
                               self.cfg.GPU)

        same_movie_type_sentence_set = self.gallery_set.get_data_from_genres(
                                        final_list)

        candidate_words_distances_list = self.search(query_set,
                                                     same_movie_type_sentence_set)
        print_log("Finish inferencing")

        return candidate_words_distances_list

    def search(self, query_set, same_type_sents):
        cand_ws_dists_list = list()
        for query, same_type_sent_list in zip(query_set, same_type_sents):
            # find word candidates by word2vec
            print_log("Searching candidates (word2vec)")
            words_dict, query_words_dict, query_tags = search_candidate_words(
                    query, same_type_sent_list, pos_dataset=self.pos_dataset)
            array_query_dict = word2vec_array(self.w2v_model, query_words_dict,
                    query_mode=True)
            tag_words = word2vec_array(self.w2v_model, words_dict)
            cand_ws_dists = eval_distance(query, query_tags, array_query_dict, tag_words,
                    method=self.cfg.method, only_word=self.cfg.hugging_face)

            # huggingface
            if self.cfg.hugging_face:
                print_log("Computing sentence similarity (huggingface)")
                query_words, words_dict = final_wordsandquery_dict(
                        query_words_dict, cand_ws_dists, self.cfg.n_largest[0])
                cand_ws_dists = hugging_face_distance(self.hf_model, self.hf_tokenizer, query,
                        query_tags, query_words, words_dict,
                        self.cfg.n_largest[1], self.cfg.GPU, method=self.cfg.method)

            # final candidate
            cand_ws_dists_list.append(cand_ws_dists)

        return cand_ws_dists_list

    def evaluate(self, tags):
        # load testing set
        test_set = np.array([[sentence, tags]
            for sentences in self.gallery_set.imdbID_to_data.values()
            for sentence in sentences if sentence])
        test_loader = [test_set[idx: min(idx + self.cfg.eval_batch_size, len(test_set))]
                for idx in range(0, len(test_set[:min(self.cfg.eval_max_sentences, len(test_set))]),
                    self.cfg.eval_batch_size)]

        # load ground truth
        with open(self.cfg.eval_ground_truth_path, newline='') as jsonfile:
            ground_truth_dict = json.loads(jsonfile.read())

        # evaluate
        total_diff = 0
        total_num = 0
        for i, batch_data in enumerate(test_loader):
            batch_output = self.inference([list(bd) for bd in batch_data])
            for i, b_output in enumerate(batch_output):
                for tag, bo in b_output.items():
                    for query_word, word_distances in bo.items():
                        if query_word in ground_truth_dict:
                            diff, total = compute([wd[1] for wd in word_distances[:min(self.cfg.eval_largest, len(word_distances))]],
                                    ground_truth_dict[query_word]['token_str'],
                                    num=self.cfg.eval_largest)
                            total_diff += diff
                            total_num += total

            info = "Total diff: {:6.2f} | Total num: {:6} | Acc: {:.6f}\n".format(total_diff,
                    total_num, total_diff / total_num)
            with open(self.cfg.eval_output_path, 'a+') as fp:
                fp.write(info)

        accuracy = total_diff / total_num

        return accuracy, total_diff, total_num
