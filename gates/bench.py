import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on sys.path (run from anywhere)
import io, os, sys, time, contextlib, collections
import numpy as np, torch, rlcard
from rlcard.agents import NFSPAgent, RandomAgent
from rlcard.utils import set_seed, reorganize
from rlcard.games.gin_rummy.utils.action_event import KnockAction, GinAction
sys.argv=["x"]
import train_gin_rummy as T
class A: seed=42; reward="points"; num_episodes=50000; load_model=None

# --- random baseline: outcome distribution + zero-sum ---
set_seed(42)
env = T.make_env(A)
env.set_agents([RandomAgent(num_actions=env.num_actions) for _ in range(2)])
oc = collections.Counter(); sums=[]; N=400
t0=time.time()
for _ in range(N):
    _, p = env.run(is_training=False)
    sums.append(abs(p[0]+p[1]))
    a = env.game.round.going_out_action
    oc["gin" if isinstance(a,GinAction) else "knock" if isinstance(a,KnockAction) else "dead"] += 1
print(f"RANDOM: {N} games {time.time()-t0:.1f}s | {dict(oc)} | max|p0+p1|={max(sums):.8f}", flush=True)

def bench(dev, tr_every, q_tr_every, M=40):
    set_seed(1); e = T.make_env(A)
    d = torch.device(dev)
    ags=[NFSPAgent(num_actions=e.num_actions, state_shape=e.state_shape[0],
         hidden_layers_sizes=[512,512], q_mlp_layers=[512,512],
         reservoir_buffer_capacity=int(2e5), q_replay_memory_size=int(2e5),
         q_epsilon_decay_steps=50000, rl_learning_rate=1e-4, sl_learning_rate=1e-4,
         train_every=tr_every, q_train_every=q_tr_every, device=d) for _ in range(2)]
    e.set_agents(ags)
    buf=io.StringIO()
    # warmup 15 eps (fills replay init), then time 40
    with contextlib.redirect_stdout(buf):
        for w in range(15+M):
            if w==15: t0=time.time()
            for a_ in ags: a_.sample_episode_policy()
            tr,p = e.run(is_training=True); tr = reorganize(tr,p)
            for a_,t_ in zip(ags,tr):
                for ts in t_: a_.feed(ts)
    dt=time.time()-t0
    print(f"{dev:5s} train_every={tr_every} q_train_every={q_tr_every}: {M/dt:5.2f} eps/s "
          f"-> 50k={50000/(M/dt)/3600:5.2f}h  100k={100000/(M/dt)/3600:5.2f}h", flush=True)
    return M/dt

bench('cuda',1,1)
bench('cuda',4,4)
bench('cuda',8,8)
bench('cpu',1,1)
bench('cpu',4,4)
