import collections
import itertools
import random
import lm_eval.metrics
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s | %(name)-7s | %(levelname)-8s: %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate(
    lm,
    task_dict,
    provide_description,
    num_fewshot,
    limit,
    bootstrap_iters=100000,
    filter_shots=True,
    serialize_predictions=False,
    **kwargs,
):
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces
    # GEO TODO: I have the impression that a lot of data content is duplicated in many structures
    #  (task, task_docs, reqs, requests, request_origin). Converting everything to HF dataset objects may be a good alternative

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    # if we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger memory,
    # we can always modify this plumbing to support that, but i didn't want to include it just yet because overengineering is bad
    # (or we could make it write the requests to disk and then read them back out again - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable

    docs = {}

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
        elif task.has_validation_docs():
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError(
                f"Task {task_name} has neither test docs nor validation docs, please verify data is properly configured for this task."
            )

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        task_docs = list(
            zip(range(len(task_docs)), task_docs)
        )  # use original sample order as ID for evaluation samples
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)

        logger.info("Found {} {} documents ...".format(len(task_docs), task_name))
        logger.info("Building requests for '{}' ...".format(task_name))

        # GEO: Maybe reqs = map(lambda x: task.construct_requests(x, task.fewshot_context(x)), itertools.islice(task_docs, 0, limit))
        for doc_id, doc in itertools.islice(task_docs, 0, limit):
            # NOTE: shot_ids and doc_ids are not global within the entire dataset, they are valid and unique within their respective sets:
            # i.e. for shot_ids usually this is the training set, for doc_ids usually this is the validation or test set.
            # The user is supposed to know which sets are used to draw shots from and which to evaluate on.
            shot_ids, ctx = task.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
                rnd=rnd,
                filter_shot_examples=filter_shots,
                **kwargs,
            )
            if isinstance(doc, dict):
                doc["doc_id"] = doc_id
                doc["shot_ids"] = shot_ids
            docs[(task_name, doc_id)] = doc
            reqs = task.construct_requests(
                doc, ctx
            )  # GEO: this is a tuple, like (ll_true, ll_false)
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.type].append(
                    req
                )  # key(s) are 'loglikelihood', etc. Each is associated with a list of Request objects, which contain (context_str, candidate_str)
                # i: index in requests for a single task instance. Each doc has as many requests as multiple choice questions.
                # doc_id: unique id that we can get back to a doc using `docs`. Just an index corresponding to the order of app. in `task_docs`
                # GEO: TODO: does it really need the `doc`? is this list necessary?
                requests_origin[req.type].append(
                    (i, task_name, doc, doc_id)
                )  # key(s) are 'loglikelihood', etc.

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)  # GEO: not a Queue though...

    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple seperate LM requests for multiple Requests differing
        # only in index. We could implement some kind of caching, but that would be more of a bandaid
        # solution. we could also implement some kind of autogrouping here; they should end up next to each other.
        # reqs is a list of request objects, as many as the samples * (num. possibile answers)
        logger.info("Running {} {} requests ...".format(len(reqs), reqtype))
        start_time = time.time()
        resps = getattr(lm, reqtype)(
            [req.args for req in reqs]
        )  # GEO: call to model for processing. (Maybe can be replaced by batching function)
        logger.info("Done in {:.3f} s".format(time.time() - start_time))
        if lm.can_access_output():
            resps = [
                x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
            ]  # list of loglikelihoods (floats)
        else:
            resps = [None] * len(reqs)
        logger.debug("Putting results in a queue for metric calculation ...")
        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append(
                (i, resp)
            )  # depending on task, for each (task, doc_id) can contain e.g. [(0, loglikelihood0), (1, loglikelihood1)]

    vals = collections.defaultdict(list)
    serialized_output = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    if lm.can_access_output():
        logger.debug("Calculating individual metrics ...")
        for (task_name, doc_id), responses in process_res_queue.items():
            responses.sort(key=lambda x: x[0])  # this sorts by class of answer, i.e. 0, 1, ...
            responses = [x[1] for x in responses]  # calculated loglikelihood for each class

            task = task_dict[task_name]
            doc = docs[(task_name, doc_id)]

            metrics = task.process_results(doc, responses)
            for metric, value in metrics.items():
                vals[(task_name, metric)].append(value)
            if hasattr(task, "serialize_results") and serialize_predictions:
                output = task.serialize_results(doc, responses)
                output["metrics"] = metrics
                serialized_output[task_name].append(output)

        # aggregate results
        logger.info("Aggregating metrics ...")
        for (task_name, metric), items in vals.items():
            task = task_dict[task_name]
            results[task_name][metric] = task.aggregation()[metric](items)

            stderr = lm_eval.metrics.stderr_for_metric(
                task.aggregation()[metric], bootstrap_iters=bootstrap_iters
            )
            if stderr is not None:
                results[task_name][metric + "_stderr"] = stderr(items)

    return_dict = {"results": results, "versions": versions}
    # NOTE(GEO): consider returning only the IDs of samples and corresponding predictions.
    # All other information can be looked up in post-processing based on ID. This will reduce storage and I/O operations.
    if serialize_predictions:
        return_dict["output"] = serialized_output

    return return_dict
