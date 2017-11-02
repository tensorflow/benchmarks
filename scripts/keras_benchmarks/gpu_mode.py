import cntk

def cntk_gpu_mode_config(model, num_samples):
    #create a CNTK distributed trainer
    model.model._make_train_function()
    trainer = model.model.train_function.trainer
    assert (trainer is not None), "Cannot find a trainer in Keras Model!"
    learner_no = len(trainer.parameter_learners)
    assert (learner_no > 0), "No learner in the trainer."
    if(learner_no > 1):
      warnings.warn("Unexpected multiple learners in a trainer.")
    learner = trainer.parameter_learners[0]
    dist_learner = cntk.train.distributed.data_parallel_distributed_learner(learner,num_quantization_bits=32,distributed_after=0)
    model.model.train_function.trainer = cntk.trainer.Trainer(
        trainer.model, [trainer.loss_function, trainer.evaluation_function], [dist_learner])

    rank = cntk.Communicator.rank()
    workers = cntk.Communicator.num_workers()
    if (workers == 1):
      warnings.warn("Only one worker is found.")
    total_items = num_samples
    start = rank*total_items//workers
    end = min((rank+1)*total_items//workers, total_items)
    return start,end

