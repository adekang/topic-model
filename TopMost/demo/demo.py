from topmost.data import download_dataset
import topmost

device = "cuda" # or "cpu"
dataset_dir = "./datasets/Wikitext-103"

dataset = topmost.data.BasicDataset(dataset_dir, read_labels=False, device=device)
model = topmost.models.ECRTM(dataset.vocab_size, pretrained_WE=dataset.pretrained_WE)
# model = topmost.models.ETM(dataset.vocab_size, pretrained_WE=dataset.pretrained_WE)
model = model.to(device)
# create a trainer
trainer = topmost.trainers.BasicTrainer(model, dataset, verbose=True)

# train the model
top_words, train_theta = trainer.train()
########################### Evaluate ####################################
# 获取训练集和测试集的theta (doc-topic分布)
train_theta, test_theta = trainer.export_theta()

# evaluate topic coherence
# refer to https://github.com/BobXWu/ECRTM

# evaluate topic diversity
TD = topmost.evaluations.compute_topic_diversity(top_words)
print(f"TD: {TD:.5f}")

# evaluate clustering
results = topmost.evaluations.evaluate_clustering(test_theta, dataset.test_labels)
print(results)

# evaluate classification
results = topmost.evaluations.evaluate_classification(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
print(results)