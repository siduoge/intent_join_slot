import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

############## mode_param ################
tf.app.flags.DEFINE_integer('word_embedding_dim', 64, 'dimension of word embedding')#118
tf.app.flags.DEFINE_integer('word_vocab_size', 724, 'size of  word')# atis
tf.app.flags.DEFINE_integer('batch_size', 16, 'number of example per batch') #512
tf.app.flags.DEFINE_integer('n_hidden', 128, 'number of hidden unit')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')#0.0005
tf.app.flags.DEFINE_float('prop', 0.5, 'trade off')#0.0005
tf.app.flags.DEFINE_integer('intent_classes', 22, 'number of distinct class') # atis
tf.app.flags.DEFINE_integer('slot_tag_size', 124, 'number of distinct slots') # atis
tf.app.flags.DEFINE_integer('max_len', 50, 'max number of tokens per sentence')
# 10
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'dropout keep prob')
tf.app.flags.DEFINE_float('smooth', 0.001, 'label smoothing')
tf.app.flags.DEFINE_float('gamma', 2, 'focal loss gamma')
tf.app.flags.DEFINE_integer('early_stopping', 5, 'the number of early stopping epoch')

#################cnn ###################
tf.app.flags.DEFINE_integer('conv_filters',128 ,'conv_filters')
tf.app.flags.DEFINE_integer('conv_kernel_size',5 ,'conv_kernel_size')
tf.app.flags.DEFINE_integer('conv_strides',1 ,'conv_strides')
tf.app.flags.DEFINE_integer('pool_size',1 ,'pool_size')
tf.app.flags.DEFINE_integer('pool_strides',1 ,'pool_strides')

#################lstm_atten ###################
tf.app.flags.DEFINE_integer('bilstm_hidden_dim',128 ,'bilstm_hidden')
tf.app.flags.DEFINE_integer('seq_hidden_dim',128 ,'seq_hidden')
tf.app.flags.DEFINE_integer('slot_embedding_dim',64 ,'slot_embedding_dim')

#################bert ###################
tf.app.flags.DEFINE_string('bert_vocab', 'bert_model/vocab', 'bert_vocab')
tf.app.flags.DEFINE_string('bert_config_json', 'bert_model/bert_config.json', 'bert_config_json')
tf.app.flags.DEFINE_string('init_checkpoint', 'bert_model/bert_model.ckpt', 'init_checkpoint')
#### transformer
tf.app.flags.DEFINE_integer('num_blocks',1 ,'while times')
tf.app.flags.DEFINE_integer('num_heads',8 ,'num_heads')
tf.app.flags.DEFINE_integer('d_ff',2048,'hidden in multi-atten')
tf.app.flags.DEFINE_integer('trans_hidden_dim1',128,'encoder outut hidden dim1')
tf.app.flags.DEFINE_integer('trans_hidden_dim2',128,'encoder out hidden dim2')

tf.app.flags.DEFINE_string('train_intent', 'data/atis/train/label', 'training intent file')
tf.app.flags.DEFINE_string('train_data', 'data/atis/train/seq.in', 'training query file')
tf.app.flags.DEFINE_string('train_slot', 'data/atis/train/seq.out', 'training slot file') 
tf.app.flags.DEFINE_string('dev_slot', 'data/atis/valid/seq.out', 'validating slot file') 
tf.app.flags.DEFINE_string('dev_intent', 'data/atis/valid/label', 'validating intent file') 
tf.app.flags.DEFINE_string('dev_data', 'data/atis/valid/seq.in', 'validating query file') 
tf.app.flags.DEFINE_string('test_slot', 'data/atis/test/seq.out', 'testing slot file') 
tf.app.flags.DEFINE_string('test_intent', 'data/atis/test/label', 'testing intent file') 
tf.app.flags.DEFINE_string('test_data', 'data/atis/test/seq.in', 'testing query file') 
tf.app.flags.DEFINE_string('out_dir', 'show_train', 'validating file') #eval/source/xu_all_traffic_no_dbp, test_data
tf.app.flags.DEFINE_string('word_vocab', 'data/dic/atis_word_vocab', 'word2idx file')
tf.app.flags.DEFINE_string('intent_vocab', 'data/dic/atis_intent_vocab', 'intent2idx file')
tf.app.flags.DEFINE_string('slot_vocab', 'data/dic/atis_slot_vocab', 'slot2idx file')
tf.app.flags.DEFINE_string('model', 'bilstm_crf', 'model choose')

tf.flags.DEFINE_integer("gpu_id", 1, "tf gpu id")
tf.flags.DEFINE_integer("topk", 4, "top k")
tf.flags.DEFINE_integer("num_gpu", 2, "tf num_gpu")
tf.flags.DEFINE_integer("num_pool", 10, "tf multi_proccser")
tf.flags.DEFINE_integer("num_epochs", 30, "Evaluate model on dev set after this many steps (default: 100)")

tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_boolean("bn_flag", True, "batch normal train")
tf.flags.DEFINE_string("export_dir", "export/show", "Checkpoint directory from training run")
tf.flags.DEFINE_string("checkpoint_dir", "runs/show_model/checkpoints", "Checkpoint directory from training run") #runs/1576667752/checkpoints
