""" CNTK gpu config required for running keras models in multi gpu mode."""
import cntk

def cntk_gpu_mode_config(model, num_samples):
    """Sets up a distributed trainer for keras models using CNTK backend
        in multi gpu mode.

    :param model: Keras model instance
    :param num_samples: total number of input training samples that will be
                        distributed across gpus for processing.
    :return: the start and end indices of the data that a given gpu will process.
    """
    # create a CNTK distributed trainer
    model.model._make_train_function()
    trainer = model.model.train_function.trainer
    learner_no = len(trainer.parameter_learners)
    if learner_no < 1:
        raise ValueError("No learner in the trainer.")
    if learner_no > 1:
      warnings.warn("Unexpected multiple learners in a trainer.")
    learner = trainer.parameter_learners[0]
    dist_learner = cntk.train.distributed.\
        data_parallel_distributed_learner(
            learner, num_quantization_bits=32, distributed_after=0)
    model.model.train_function.trainer = cntk.trainer.Trainer(
        trainer.model, [trainer.loss_function,
                        trainer.evaluation_function], [dist_learner])

    rank = cntk.Communicator.rank()
    workers = cntk.Communicator.num_workers()
    if workers == 1:
      warnings.warn("Only one worker is found.")
    total_items = num_samples
    start = rank * total_items // workers
    end = min((rank+1) * total_items // workers, total_items)
    return start, end

