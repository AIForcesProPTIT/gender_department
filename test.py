if __name__ == "__main__":
    from preprocess.static_features import Feature
    from model.layer.featureEmbed import FeatureRep
    import torch
    feature = Feature("./resources/features/feature_config.json", True)
    featrep = FeatureRep(feature)
    print(featrep)
    sample_input = {}
    for feat_key in feature.feature_keys:
        vocab_size = len(feature.feature_infos[feat_key]["label"])
        sample_input[feat_key]=torch.randint(low=0, high=vocab_size, size=(1, 5), dtype=torch.long)
    outs = featrep(sample_input)
    print(outs.shape)