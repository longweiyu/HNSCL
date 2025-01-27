from model import CLBert
from init_parameter import init_model
from dataloader import Data
from pretrain import DomainPretrainModelManager
from utils.tools import *
from utils.memory import MemoryBank, fill_memory_bank
from utils.neighbor_dataset import NeighborsDataset
from utils.logging_utils import setup_logger
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import json
from collections import Counter
from utils.file_utils import get_run_filename
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils.file_utils import get_run_filename, FileType
from enum import Enum
from dataclasses import dataclass
from utils.tools import plot_clustering_results  # Update the import statement
from utils.file_utils import ClusteringUtils
from utils.file_utils import ExperimentMeta
import pandas as pd

class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        # Setup logger
        self.logger, _ = setup_logger(
            name='HNSCL',
            dataset=args.dataset, 
            known_cls_ratio=args.known_cls_ratio, 
            labeled_ratio=args.labeled_ratio,
            log_file=manager_pretrain.log_file if not args.disable_pretrain else None,
            timestamp=manager_pretrain.run_id if not args.disable_pretrain else time.strftime("%Y%m%d_%H%M%S")
        )
        
        # Generate run IDs (if there is no pre-training, create a new one; If there is pre-training, use the same)
        self.run_id = manager_pretrain.run_id if not args.disable_pretrain else time.strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Initialized run with ID: {self.run_id}")
        
        set_seed(args.seed)
        self.logger.info(f"Set random seed to {args.seed}")
        
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device} (n_gpu: {n_gpu})")
        
        self.num_labels = data.num_labels
        # Cl
        self.model = CLBert(args.bert_model, device=self.device)
        self.logger.info(f"Initialized HNSCL model with {self.num_labels} labels")

        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        
        if not args.disable_pretrain:
            self.pretrained_model = pretrained_model
            self.load_pretrained_model()#load the weights of pretrain model
            self.logger.info("Loaded pretrained model")
        
        self.num_train_optimization_steps = int(len(data.train_semi_dataset) / args.train_batch_size) * args.num_train_epochs
        self.logger.info(f"Total optimization steps: {self.num_train_optimization_steps}")
        
        self.optimizer, self.scheduler = self.get_optimizer(args)
        self.logger.info("Initialized optimizer and scheduler")
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)
        self.logger.info(f"Initialized tokenizer and view generator with RTR prob: {args.rtr_prob}")
    
    def get_neighbor_dataset(self, args, data, indices):
        """convert indices to dataset"""
        dataset = NeighborsDataset(data.train_semi_dataset, indices)
        self.train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    
    def get_adj_matrix(self, args, inds, neighbors, targets):
        """get adjacency matrix"""
        adj_matrix = torch.zeros(inds.shape[0], inds.shape[0])
        for x1, n in enumerate(neighbors):
            adj_matrix[x1][x1] = 1
            for x2, j in enumerate(inds):
                if j in n:
                    adj_matrix[x1][x2] = 1 # if in neighbors
                if (targets[x1] == targets[x2]) and (targets[x1]>0) and (targets[x2]>0):
                    adj_matrix[x1][x2] = 1 # if same labels
                    # this is useful only when both have labels
        return adj_matrix
    
    def get_neighbors_hnsw(self, args, data):
    # use HNSW to find neighbor indexes
        try:
            self.logger.info("Start building the HNSW memory space...")
            memory_bank = MemoryBank(
                len(data.train_semi_dataset),
                args.feat_dim,
                len(data.all_label_list),
                0.1
            )

            self.logger.info("Filling memory space...")
            fill_memory_bank(data.train_semi_dataloader, self.model, memory_bank)
            self.logger.info(f"Run HNSW searching (k={args.topk})...")
            indices = memory_bank.mine_nearest_neighbors_hnsw(
                args.topk,
                calculate_accuracy=False
            )

            expected_shape = (len(data.train_semi_dataset), args.topk + 1)
            if indices.shape != expected_shape:
                raise ValueError(f"Index shape error: Expectation {expected_shape}, but now {indices.shape}")

            self.logger.info("HNSW searching finished")
            return indices

        except Exception as e:
            self.logger.error(f"HNSW searching fail: {str(e)}")
            raise e



    def evaluation(self, args, data, save_results=True):
        exp_meta = ExperimentMeta(
            dataset=args.dataset,
            known_ratio=args.known_cls_ratio,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed,
            timestamp=self.run_id
        )
        
        self.logger.info('Starting clustering and evaluation on test set')
        self.logger.info(f"Performing K-means clustering with {self.num_labels} clusters...")
        
        features, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        features_np = features.cpu().numpy()
        y_true = labels.cpu().numpy()
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # get original text
        texts = []
        for batch in data.test_dataloader:
            batch_texts = tokenizer.batch_decode(batch[0], skip_special_tokens=True)
            texts.extend(batch_texts)
        
        # start clustering
        self.logger.info(f"Performing K-means clustering with {self.num_labels} clusters...")
        km = KMeans(n_clusters=self.num_labels, n_init=10, random_state=args.seed)
        y_pred = km.fit_predict(features_np)
        
        # save clustering images
        ClusteringUtils.save_clustering_visualization(
                exp_meta=exp_meta,
                features=features_np,
                y_true=y_true,
                y_pred=y_pred,
                logger=self.logger
        )
        
        # save detailed lustering images
        ClusteringUtils.save_detailed_clustering(
            exp_meta=exp_meta,
            features=features_np,
            y_pred=y_pred,
            logger=self.logger,
            title=f'Testset Clustering Distribution'
        )
        
        ClusteringUtils.save_clustering_log(
            exp_meta=exp_meta,
            texts=texts,
            y_pred=y_pred,
            num_clusters=self.num_labels,
            logger=self.logger
        )
        
        results = clustering_score(y_true, y_pred)
        ClusteringUtils.log_clustering_metrics(results, self.logger)
        
        self.test_results = results
        if save_results:
            self.save_results(args)
        
        def save_prediction_results(exp_meta, texts, y_true, y_pred, logger):
            try:
                results_df = pd.DataFrame({
                    'text': texts,
                    'label': y_true,
                    'predicted_label': y_pred
                })
                
                pred_file = get_run_filename(
                    exp_meta,
                    FileType.PREDICTION,
                    stage='test_predictions',
                    logger=logger
                )
                results_df.to_csv(pred_file, sep='\t', index=False)
                logger.info(f"Saving prediction results to: {pred_file}")
                
            except Exception as e:
                logger.error(f"Errors saving prediction results: {str(e)}")
                raise e
        
        save_prediction_results(
            exp_meta=exp_meta,
            texts=texts,
            y_true=y_true,
            y_pred=y_pred,
            logger=self.logger
        )
        
        return results

    def train(self, args, data):
        self.logger.info('=====start training====')
        if isinstance(self.model, nn.DataParallel):
            criterion = self.model.module.loss_cl
        else:
            criterion = self.model.loss_cl
        
        # load neighbors for the first epoch
        self.logger.info("Getting initial neighbors...")
        indices = self.get_neighbors_hnsw(args, data)
        self.get_neighbor_dataset(args, data, indices)
        self.logger.info("Initialized neighbor dataset")

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for batch in tqdm(self.train_dataloader, desc="Iteration"):
                # 1. load data
                anchor = tuple(t.to(self.device) for t in batch["anchor"]) # anchor data
                neighbor = tuple(t.to(self.device) for t in batch["neighbor"]) # neighbor data
                pos_neighbors = batch["possible_neighbors"] # all possible neighbor inds for anchor
                data_inds = batch["index"] # neighbor data ind

                # 2. get adjacency matrix
                adjacency = self.get_adj_matrix(args, data_inds, pos_neighbors, batch["target"]) # (bz,bz)

                # 3. get augmentations
                X_an = {"input_ids":self.generator.random_token_replace(anchor[0].cpu()).to(self.device), "attention_mask":anchor[1], "token_type_ids":anchor[2]}
                X_ng = {"input_ids":self.generator.random_token_replace(neighbor[0].cpu()).to(self.device), "attention_mask":neighbor[1], "token_type_ids":neighbor[2]}
                # 4. compute loss and update parameters
                with torch.set_grad_enabled(True):
                    # Pass augmented anchor and neighbor through the model to get embeddings
                    f_pos = torch.stack([self.model(X_an)["features"], self.model(X_ng)["features"]], dim=1)
                    loss = criterion(f_pos, mask=adjacency, temperature=args.temp)
                    tr_loss += loss.item()
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    nb_tr_examples += anchor[0].size(0)
                    nb_tr_steps += 1
            
            # Log epoch results
            loss = tr_loss / nb_tr_steps
            self.logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")
            self.logger.info(f"train_loss {loss}")
            
            # Update neighbors for next epoch
            if ((epoch + 1) % args.update_per_epoch) == 0:
                self.logger.info("Updating neighbors for next epoch...")
                indices = self.get_neighbors_hnsw(args, data)
                self.get_neighbor_dataset(args, data, indices)

        self.logger.info('=====end training====')

    def get_optimizer(self, args):
        num_warmup_steps = int(args.warmup_proportion*self.num_train_optimization_steps)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=self.num_train_optimization_steps)
        return optimizer, scheduler
    
    def load_pretrained_model(self):
        """load the backbone of pretrained model"""
        if isinstance(self.pretrained_model, nn.DataParallel):
            pretrained_dict = self.pretrained_model.module.backbone.state_dict()
        else:
            pretrained_dict = self.pretrained_model.backbone.state_dict()
             # 1.4 load to backbone of CLBert
        if isinstance(self.model, nn.DataParallel):
            self.model.module.backbone.load_state_dict(pretrained_dict, strict=False)
        else:
            #directly load the pre-trained weights into the model’s backbone
            self.model.backbone.load_state_dict(pretrained_dict, strict=False)

    def get_features_labels(self, dataloader, model, args):
        """get features and labels"""
        self.logger.info("Starting feature extraction...")
        model.eval()
        
        # initialize empty tensors without specifying dimensions
        total_features = None
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)

        try:
            for batch in tqdm(dataloader, desc="Extracting representation"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                X = {"input_ids":input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
                
                with torch.no_grad():
                    outputs = model(X)
                    
                    # get features
                    if isinstance(outputs, dict):
                        feature = outputs.get("features", None)
                        if feature is None:
                            feature = outputs.get("pooler_output", outputs.get("last_hidden_state")[:, 0, :])
                    else:
                        feature = outputs[1] if isinstance(outputs, tuple) else outputs
                    
                    if total_features is None:
                        self.logger.info(f"Feature shape from first batch: {feature.shape}")
                        total_features = feature
                    else:
                        #make sure dims matche
                        if feature.shape[1] != total_features.shape[1]:
                            self.logger.error(f"Dimension mismatch: existing {total_features.shape[1]} vs new {feature.shape[1]}")
                            raise ValueError(f"Feature dimensions don't match: {total_features.shape[1]} vs {feature.shape[1]}")
                        total_features = torch.cat((total_features, feature))
                    
                    total_labels = torch.cat((total_labels, label_ids))

        except Exception as e:
            self.logger.error(f"Error during feature extraction: {str(e)}")
            raise e

        self.logger.info(f"Final feature tensor shape: {total_features.shape}")
        self.logger.info(f"Final labels tensor shape: {total_labels.shape}")
        
        return total_features, total_labels

    def save_results(self, args):
        exp_meta = ExperimentMeta(
            dataset=args.dataset,
            known_ratio=args.known_cls_ratio,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed,
            timestamp=self.run_id
        )
        
        # prepare result parameters
        results_data = {
            'ACC': [self.test_results['ACC']],
            'ARI': [self.test_results['ARI']],
            'NMI': [self.test_results['NMI']],
            'dataset': [args.dataset],
            'method': [args.method],
            'known_cls_ratio': [args.known_cls_ratio],
            'labeled_ratio': [args.labeled_ratio],
            'topk': [args.topk],
            'seed': [args.seed],
            'num_pretrain_epochs': [args.num_pretrain_epochs],
            'num_train_epochs': [args.num_train_epochs],
            'pretrain_batch_size': [args.pretrain_batch_size],
            'train_batch_size': [args.train_batch_size]
        }
        new_results = pd.DataFrame(results_data)
        
        results_path = get_run_filename(
            exp_meta=exp_meta,
            file_type=FileType.RESULTS,
            stage='experiment',
            logger=self.logger
        )
        
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        if os.path.exists(results_path):
            existing_results = pd.read_csv(results_path)
            combined_results = pd.concat([existing_results, new_results], ignore_index=True)
        else:
            combined_results = new_results
            
        combined_results.to_csv(results_path, index=False)
        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info(f"Current results:\n{new_results}")

    def save_model(self, args):
        self.logger.info('=====Saving Model ...')
        exp_meta = ExperimentMeta(
            dataset=args.dataset,
            known_ratio=args.known_cls_ratio,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed,
            timestamp=self.run_id
        )
        
        model_filename = get_run_filename(
            exp_meta=exp_meta,
            file_type=FileType.MODEL,
            stage='experiment',
            logger=self.logger,
            get_name_only=True
        )
        
        model_path = os.path.join('models', model_filename)
        os.makedirs('models', exist_ok=True)
        
        self.model.save_backbone(model_path)
        self.logger.info(f"=====Saved trained model to {model_path}")
        self.logger.info('=====Saving Model Finished...')

if __name__ == '__main__':
    # 1. initialize parameters
    print('Data and Parameters Initialization...')
    parser = init_model() 
    args = parser.parse_args()
    print(args)
    #generate global run id
    run_id = time.strftime("%Y%m%d_%H%M%S")
    # 2. set log files
    exp_meta = ExperimentMeta(
        dataset=args.dataset,
        known_ratio=args.known_cls_ratio,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
        timestamp=run_id
    )
    
    # initialize pretraining logger
    pretrain_logger, log_file = setup_logger(
        name='pretrain',
        dataset=args.dataset,
        known_cls_ratio=args.known_cls_ratio,
        labeled_ratio=args.labeled_ratio,
        timestamp=run_id
    )
    
    # 2. if run pretraining
    if args.known_cls_ratio == 0:
        args.disable_pretrain = True
    else:
        args.disable_pretrain = False
    data = Data(args)

    # 3.1pre-training
    print('=====Pre-training begin...')
    manager_pretrain = DomainPretrainModelManager(args, data)
    manager_pretrain.run_id = run_id
    manager_pretrain.train(args, data)
    print('=====Pre-training finished=====')
    
    # 3.2 start training 
    print('=====Training initialization start=====')
    model_manager = ModelManager(args, data, manager_pretrain.model)
    model_manager.run_id = manager_pretrain.run_id

    print('=====Initialization finished=====')
    print('Training begin...')
    model_manager.train(args, data)
    print('Training finished!')

    # 4. final evaluation
    print('Final Evaluation begin...')
    model_manager.evaluation(args, data)
    print('Final Evaluation finished!')

    # 5. save model
    if args.save_model_path:
        model_manager.save_model(args)
    print("=====Finished!=====")
