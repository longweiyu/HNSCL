from utils.tools import *
from model import BertForModel

class DomainPretrainModelManager:
    def evaluation(self, args, data):
        self.model.eval()  # set model to evaluation mode
        #  (Ground-truth labels for all samples)  # [2, 1, 3]
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)
        #(Raw prediction scores for all samples)
        # [[1.0, 2.5, 0.3, -1.2],
        #  [0.4, 3.1, 1.2, 0.0],
        #  [-0.5, 0.3, 2.0, 1.7]]
        # iterate over validation set
        for batch in tqdm(data.eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            # forward pass with closing grad update
            with torch.set_grad_enabled(False):
                # input data of each batch of validation set
                X = {"input_ids": input_ids, 
                     "attention_mask": input_mask, 
                     "token_type_ids": segment_ids}
                #prediction results
                logits = self.model(X)["logits"]
                
                # concat ground-truth labels of data in each batch
                total_labels = torch.cat((total_labels, label_ids))
                # concat prediction labels of data in each batch
                total_logits = torch.cat((total_logits, logits))
        
        # total_preds:prediction labels
        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        accuracy_score = round(accuracy_score(y_true, y_pred) * 100, 2)
        
        return accuracy_score
        
    def train(self, args, data):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # Tracks how many consecutive epochs the validation score hasn't improved (used for early stopping).
        wait = 0
        #Stores the best-performing model based on validation accuracy.
        best_model = None
        # Iterator for the semi-supervised data loader for MLM.
        mlm_iter = iter(data.train_semi_dataloader)
        
        for epoch in trange(int(args.num_pretrain_epochs)):
            # 1. train one epoch
            self.model.train()
            sum_train_loss = 0
            num_train_examples, nb_tr_steps = 0, 0
            if args.known_cls_ratio > 0:
                dataloader = data.train_labeled_dataloader
            else:
                dataloader = data.train_semi_dataloader
                
            for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                try:
                    mlm_batch = next(mlm_iter)
                    mlm_batch = tuple(t.to(self.device) for t in mlm_batch)
                    mlm_input_ids, mlm_input_mask, mlm_segment_ids, _ = mlm_batch
                except StopIteration:
                    mlm_iter = iter(data.train_semi_dataloader)
                    mlm_batch = next(mlm_iter)
                    mlm_batch = tuple(t.to(self.device) for t in mlm_batch)
                    mlm_input_ids, mlm_input_mask, mlm_segment_ids, _ = mlm_batch
                    
                X_mlm = {"input_ids":mlm_input_ids, "attention_mask": mlm_input_mask, "token_type_ids": mlm_segment_ids}
                mask_ids, mask_lb = mask_tokens(X_mlm['input_ids'].cpu(), tokenizer)
                X_mlm["input_ids"] = mask_ids.to(self.device)

                # 3. compute loss and update parameters
                with torch.set_grad_enabled(True):
                    if args.known_cls_ratio > 0:
                        logits = self.model(X)["logits"]
                        loss_src = self.model.loss_ce(logits, label_ids)
                    else:
                        loss_src = 0
                    loss_mlm = self.model.mlmForward(X_mlm, mask_lb.to(self.device))
                    if args.known_cls_ratio > 0:
                        loss_sum = loss_src + loss_mlm
                    else:
                        loss_sum = loss_mlm
                    loss_sum.backward()
                    # prevent exploding gradients and stabilize training
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    sum_train_loss += loss_sum.item()
                    #Updates model parameters using the optimizer
                    self.optimizer.step()
                    #adjusts the learning rate with the scheduler.
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    num_train_examples += input_ids.size(0)
                    nb_tr_steps += 1
                
            avg_loss = sum_train_loss / nb_tr_steps
            print('avg_loss', avg_loss)
            
            # evaluation and early stopping
            if args.known_cls_ratio > 0:
                self.model.eval()
                eval_score = self.evaluation(args, data)
                print('score', eval_score)
                
                if eval_score > self.best_eval_score:
                    best_model = copy.deepcopy(self.model)
                    wait = 0
                    self.best_eval_score = eval_score
                else:
                    wait += 1
                    if wait >= args.wait_patient:
                        break
            else:
                if avg_loss < self.best_eval_score or self.best_eval_score == 0:
                    best_model = copy.deepcopy(self.model)
                    self.best_eval_score = avg_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= args.wait_patient:
                        break
                
        self.model = best_model
        
    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion*self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_pre)
        # scheduler:make learning rate increase in warmup steps and then decrease to 0
        # num_train_optimization_steps: total training steps
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        # return opimizer method and learning rate scheduler
        return optimizer, scheduler
        
    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature = model(X, output_hidden_states=True)["hidden_states"]
            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))
        return total_features, total_labels
    """
    Domain-specific pre-training phase
    """
    def __init__(self, args, data):
        # set rabdom seed
        set_seed(args.seed)
        # GPU settings
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         # initialize BERT model
        self.model = BertForModel(args.bert_model, num_labels=data.n_known_cls, device=self.device)
        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        # calculate total optimization steps
        self.num_train_optimization_steps = int(
            len(data.train_labeled_examples) / args.pretrain_batch_size  # 每个epoch的步数
        ) * args.num_pretrain_epochs
        # get optimizer and scheduler
        self.optimizer, self.scheduler = self.get_optimizer(args)
        self.best_eval_score = 0