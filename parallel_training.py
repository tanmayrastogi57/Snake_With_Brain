import multiprocessing as mp
import os
import queue
import random
import traceback

import numpy as np
import torch

from agent import (
    Agent,
    BATCH_SIZE,
    TRAIN_EVERY,
    evaluate,
    seed_everything,
)
from experiment import JsonlLogger
from game import SnakeGameAI
from helper import plot_training_history


DEFAULT_POLICY_SYNC_EVERY = 250
TRANSITION_QUEUE_MULTIPLIER = 256


def _put_message(message_queue, message, stop_event):
    while not stop_event.is_set():
        try:
            message_queue.put(message, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def _apply_policy(worker_agent, policy_message):
    worker_agent.load_policy_state(policy_message['state'])
    worker_agent.n_games = policy_message['n_games']


def _drain_policy_updates(worker_agent, policy_queue):
    latest = None
    while True:
        try:
            latest = policy_queue.get_nowait()
        except queue.Empty:
            break
    if latest is not None:
        _apply_policy(worker_agent, latest)


def _environment_worker(worker_id, transition_queue, policy_queue, stop_event, seed, render):
    """Generate experience on CPU; the learner owns the shared model."""
    try:
        torch.set_num_threads(1)
        seed_everything(seed)
        worker_agent = Agent(resume=False, device='cpu')
        initial_policy = policy_queue.get()
        _apply_policy(worker_agent, initial_policy)
        game = SnakeGameAI(
            render=render,
            rng=random.Random(seed),
            window_title=f'Snake Worker {worker_id}',
        )

        try:
            while not stop_event.is_set():
                _drain_policy_updates(worker_agent, policy_queue)
                state_old = worker_agent.get_state(game)
                final_move = worker_agent.get_action(state_old)
                reward, done, score = game.play_step(final_move)
                state_new = worker_agent.get_state(game)

                if not _put_message(
                    transition_queue,
                    ('transition', worker_id, state_old, int(np.argmax(final_move)), reward, state_new, done),
                    stop_event,
                ):
                    break

                if done:
                    if not _put_message(transition_queue, ('episode', worker_id, score), stop_event):
                        break
                    game.reset()
        finally:
            game.close()
    except Exception:
        traceback.print_exc()
        stop_event.set()
        raise


def _policy_message(agent):
    return {
        'state': agent.policy_state_cpu(),
        'n_games': agent.n_games,
    }


def _publish_policy(policy_queues, agent):
    message = _policy_message(agent)
    for policy_queue in policy_queues:
        while True:
            try:
                policy_queue.get_nowait()
            except queue.Empty:
                break
        try:
            policy_queue.put_nowait(message)
        except queue.Full:
            pass


def train_parallel(
    workers,
    max_games=None,
    seed=None,
    eval_games=0,
    eval_every=0,
    log_path='runs/training.jsonl',
    resume=True,
    checkpoint_path='model/checkpoint.pth',
    device='auto',
    policy_sync_every=DEFAULT_POLICY_SYNC_EVERY,
    render_workers=False,
    plot_path='runs/shared/shared-training.png',
    plot_every=10,
):
    """Train one shared policy from N parallel environments."""
    if workers < 2:
        raise ValueError('train_parallel requires at least 2 workers')
    if policy_sync_every < 1:
        raise ValueError('policy_sync_every must be positive')
    if plot_every < 1:
        raise ValueError('plot_every must be positive')

    if seed is not None:
        seed_everything(seed)

    agent = Agent(checkpoint_path=checkpoint_path, resume=resume, device=device)
    logger = JsonlLogger(
        log_path,
        context={
            'seed': seed,
            'render': render_workers,
            'resume': resume,
            'workers': workers,
            'device': str(agent.device),
            'checkpoint_path': agent.checkpoint_path,
        },
    )
    logger.log(
        'run_started',
        max_games=max_games,
        eval_games=eval_games,
        eval_every=eval_every,
        policy_sync_every=policy_sync_every,
    )

    context = mp.get_context('spawn')
    transition_queue = context.Queue(maxsize=max(workers * TRANSITION_QUEUE_MULTIPLIER, 1024))
    policy_queues = [context.Queue(maxsize=1) for _ in range(workers)]
    stop_event = context.Event()
    processes = []
    start_games = agent.n_games
    target_games = None if max_games is None else start_games + max_games
    update_count = 0
    history = []

    _publish_policy(policy_queues, agent)
    for worker_id, policy_queue in enumerate(policy_queues):
        worker_seed = (seed if seed is not None else random.randrange(2**31)) + worker_id
        process = context.Process(
            target=_environment_worker,
            args=(worker_id, transition_queue, policy_queue, stop_event, worker_seed, render_workers),
            name=f'snake-worker-{worker_id}',
        )
        process.start()
        processes.append(process)

    try:
        while not stop_event.is_set():
            if target_games is not None and agent.n_games >= target_games:
                break

            try:
                message = transition_queue.get(timeout=0.5)
            except queue.Empty:
                failed = [process for process in processes if not process.is_alive() and process.exitcode]
                if failed:
                    raise RuntimeError(f'Parallel worker failed with exit code {failed[0].exitcode}')
                continue

            if message[0] == 'transition':
                _, _, state, action, reward, next_state, done = message
                agent.memory.add(state, action, reward, next_state, done)
                agent.frame_count += 1
                if agent.frame_count % TRAIN_EVERY == 0 and agent.train_from_replay():
                    update_count += 1
                    if update_count % policy_sync_every == 0:
                        _publish_policy(policy_queues, agent)
            elif message[0] == 'episode':
                _, worker_id, score = message
                agent.n_games += 1
                agent.record = max(agent.record, score)
                agent.total_score += score
                mean_score = agent.total_score / agent.n_games
                agent.plot_scores.append(score)
                agent.plot_mean_scores.append(mean_score)
                print(
                    f'Worker {worker_id} | Game {agent.n_games} | '
                    f'Score {score} | Record {agent.record} | '
                    f'Replay {len(agent.memory)} | Device {agent.device}'
                )
                episode_record = {
                    'worker': worker_id,
                    'game': agent.n_games,
                    'score': score,
                    'record': agent.record,
                    'mean_score': mean_score,
                    'epsilon': Agent.epsilon_for_game(agent.n_games),
                    'frame_count': agent.frame_count,
                    'replay_size': len(agent.memory),
                    'learner_updates': update_count,
                }
                history.append({'event': 'episode', **episode_record})
                logger.log(
                    'episode',
                    **episode_record,
                )

                if eval_games and (eval_every == 0 or agent.n_games % eval_every == 0):
                    evaluation = evaluate(agent, games=eval_games, seed=seed, render=False)
                    logger.log('evaluation', game=agent.n_games, **evaluation)
                    history.append({'event': 'evaluation', 'game': agent.n_games, **evaluation})
                    print(
                        f"Evaluation after game {agent.n_games}: "
                        f"mean={evaluation['mean_score']:.2f} max={evaluation['max_score']}"
                    )

                if plot_path and agent.n_games % plot_every == 0:
                    plot_training_history(history, plot_path)

                if target_games is not None and agent.n_games >= target_games:
                    break
    except KeyboardInterrupt:
        logger.log('run_finished', reason='keyboard_interrupt', game=agent.n_games, frame_count=agent.frame_count)
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        agent.save_checkpoint()
        if plot_path:
            plot_training_history(history, plot_path)
        logger.log(
            'run_finished',
            reason='completed' if target_games is not None and agent.n_games >= target_games else 'stopped',
            game=agent.n_games,
            frame_count=agent.frame_count,
            replay_size=len(agent.memory),
            learner_updates=update_count,
        )
        transition_queue.close()
        for policy_queue in policy_queues:
            policy_queue.close()

    return agent
