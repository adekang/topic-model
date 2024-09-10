from FASTopic.fastopic import FASTopics
from TopMost import topmost
from TopMost.topmost.data import BasicDataset
import numpy as np


def model_test(model, dataset, num_topics):
    docs = dataset.train_texts
    top_words, train_theta = model.fit_transform(docs)
    test_theta = model.transform(dataset.test_texts)

    assert len(top_words) == num_topics
    assert train_theta.shape[0] == len(docs)
    assert test_theta.shape[0] == len(dataset.test_texts)

    model.get_topic(0)
    TD = topmost.evaluations.compute_topic_diversity(top_words)
    print(f"TD: {TD:.5f}")

    TC = topmost.evaluations.compute_topic_coherence(dataset.train_texts, dataset.vocab, top_words)
    print(f"TC: {TC:.5f}")
    # evaluate clustering

    results = topmost.evaluations.evaluate_clustering(test_theta, dataset.test_labels)
    print(results)

    # evaluate classification
    results = topmost.evaluations.evaluate_classification(train_theta, test_theta, dataset.train_labels,
                                                          dataset.test_labels)
    print(results)


def test_models(cache_path, num_topics):
    # download_dataset("20NG", cache_path=f"{cache_path}/datasets")
    dataset = BasicDataset(f"{cache_path}/datasets/20NG", as_tensor=False, read_labels=True)

    model = FASTopics(num_topics=num_topics, epochs=3, verbose=True)
    model_test(model, dataset, num_topics)


if __name__ == '__main__':
    test_models(cache_path='.', num_topics=10)
