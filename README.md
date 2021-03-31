# 意图和槽位联合识别 intent_join_slot
我集成了很多篇paper的工作:  
intent_acc: 1, slot_f1: 95.0972   
bilstm_base: intent_acc: 0.958567,slot_f1: 91.3005    
bilstm_crf: intent_acc: 0.921613,slot_f1: 94.0527  
bilstm_attention: intent_acc: 0.966405,slot_f1: 92.0573  
bilstm_slot_gated: -  
cnn_crf: intent_acc: 0.791713,slot_f1: 85.9571 实现不出来    
bilstm_bert: intent_acc: 1, slot_f1: 95.0972  
自己用来做着练手的，效果因为数据集的原因可能有些问题，还在建设中。。。。。。  
知乎文章:https://zhuanlan.zhihu.com/p/330378804  

