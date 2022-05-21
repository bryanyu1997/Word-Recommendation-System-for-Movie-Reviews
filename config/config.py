class Config:
    # dataset
    re_pos_tags = False
    imdb_path = "./data/raw/"
    data_path = "./data/file/"
    genres_path = "./data/dict_of_genres.json"
    set_name = "total"
    pos_dataset = "./data/pos_" + set_name + ".pkl"

    # classifier
    MODEL_STATE_DICT = "weights/model_50.pth"
    N_CLASSES = 29
    THRESHOLD = 0.7
    BERT_MODEL_NAME = "bert-base-uncased"
    MAX_TOKEN_LENGTH = 512
    BATCH_SIZE = 1
    GPU = False  # torch.cuda.is_available()

    # main
    hugging_face = True
    w2v_model_name = "weights/GoogleNews-vectors-negative300.bin.gz"
    hugging_face_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # search
    method = "cosine"  # cosine, kl, l2
    n_largest = [2, 5]  # [eval_distance_numbers, hugging_face_numbers]

    # evaluate
    eval_output_path = "eval.txt"
    eval_gt_path = "./data/ground_truth/20000.json"
    eval_batch_size = 1
    eval_largest = 100
    eval_max_sentences = 500
