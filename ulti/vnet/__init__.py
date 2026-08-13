"""Value-net inputs.

``pickup`` — the deployed bidding featurizer (hand + trump → the vector every
bidding head reads) plus the suit-permutation augmentation the trainer uses.

The heads themselves are ``ulti.bidding.base_head.Head``; training them is
``ulti.pipeline.frontier_heads``.
"""
