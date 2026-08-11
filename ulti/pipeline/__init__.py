"""The frontier training pipeline — datagen → train → calibrate → audit, one path.

Born 2026-08-11 after the duri lesson: three head-training pipelines had drifted
apart (exp17's augmented trainer, exp42's bespoke duri retrain that silently
skipped augmentation, exp37's betli_real). Every future head retrain goes through
frontier_heads — there is deliberately nowhere else to train one.
"""
