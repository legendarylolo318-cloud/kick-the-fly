"""Golden for ktf_data::streamer: scripted chat through core/streamer.py's VoteBoard."""
import json, sys
sys.path.insert(0, "legacy-python")
from kickthefly.core import streamer as S

b = S.VoteBoard(S.Rules(window_s=10.0, cooldown_s=15.0, min_votes=2, allow=frozenset({"tool", "arena"})),
                {"tool": {"swatter": "SWATTER", "bomb": "BOMB", "frog_food": "FROG FOOD"}, "arena": {"pool": "pool", "lamp": "lamp"}, "surgery": {"x": "X"}})
script = [("a", "!tool swatter", 0.0), ("a", "!tool bomb", 0.4), ("b", "!tool Frog Food", 1.0), ("c", "!tool bomb", 1.2),
          ("d", "!arena pool", 2.0), ("e", "!surgery x", 2.1), ("f", "hello", 2.2), ("g", "!tool nope", 2.3), ("a", "!tool bomb", 3.0),
          ("h", "!tool bomb", 3.5), ("i", "!arena lamp", 9.9)]
out = []
for u, t, now in script:
    out.append([u, t, now, b.submit(u, t, now), [list(r) for r in b.tally()], b.state, b.remaining(now)])
res = b.tick(10.0)
ticks = [[10.0, None if res is None else [res.command, res.slug, res.label, res.votes, res.total], b.state, b.remaining(10.0)]]
for u, t, now in [("a", "!tool swatter", 12.0), ("a", "!tool swatter", 26.0), ("b", "!tool swatter", 27.0)]:
    out.append([u, t, now, b.submit(u, t, now), [list(r) for r in b.tally()], b.state, b.remaining(now)])
res = b.tick(36.0)
ticks.append([36.0, None if res is None else [res.command, res.slug, res.label, res.votes, res.total], b.state, b.remaining(36.0)])
lines = ["@a=b :bob!bob@x PRIVMSG #Chan :!tool swatter", "PING :tmi.twitch.tv", "PING", ":tmi.twitch.tv 001 justinfan1 :Welcome", ":tmi NOTICE * :Login authentication failed", "garbage", ""]
json.dump(dict(submits=out, ticks=ticks, lines=[[l, S.parse_line(l)] for l in lines],
               slugs=[[s, S.slug(s)] for s in ["  Hello, World!! ", "Frog-Food 2", "___", "ÄÖ x"]]), open("tests/golden/streamer.json", "w"), indent=1)
print("ok")
