# Streamer mode (3.0 day 3)

**Opt-in. Off every time the game starts.** Esc > **Mic and streamer** > Streamer mode. Viewers of a Twitch channel vote on what the game does next. YouTube is **not supported** (it can't be done without storing a key insecurely).

**What it connects to, shown on screen:** while it is on, a red pill says `TWITCH CHAT - reading irc.chat.twitch.tv:6697 #channel (read-only)` (host, port and channel) and the page lists the connection. It is the only network feature added for play; the other network use, older than it, is the brain view's one-time download of ten neuron skeletons from neuPrint (cached; `KICK_THE_FLY_OFFLINE=1` turns it off, and headless runs never do it). **No telemetry, ever.** It never runs in `--validate`, assays, protocols, replays, the playthrough or any test (`core/netguard.py`: headless runs switch the network off; the test suite sets `KTF_NO_NETWORK`), and it is never reopened by itself at the next launch.

How it connects (`core/streamer.py`): TLS to `irc.chat.twitch.tv` port 6697, nickname `justinfan<random digits>`, **no password and no token**, `JOIN` one channel, answer `PING`. It is **read-only**: it never sends a chat message, logs in, or stores a key. The channel name is validated (3-25 letters, digits, underscores) so nothing typed can add a second IRC command. Twitch documents the IRC interface at dev.twitch.tv/docs/chat/irc/; the anonymous `justinfan` read-only login is described in Twitch's developer-forum threads (discuss.dev.twitch.com "Anonymous connection using justinfan") but **not in the current official documentation**, so Twitch may stop allowing it; if the server refuses, the game says so and nothing else happens. (Checked by the 3.0 day 3 review: the official page gives `irc.chat.twitch.tv:6697`, `PING`/`PONG` and a login with `PASS oauth:...`, and says nothing about anonymous access; in the February 2024 forum announcement of concurrent join limits, a forum moderator wrote that justinfan "was never officially documented" and that its future under those limits was not known.)

## Voting (all GAME RULE)

| | |
|---|---|
| commands | `!tool NAME`, `!arena NAME`, `!surgery NAME` |
| the streamer's allowlist | Settings > Brain > Streamer mode: which commands count. **`!tool` is on; `!arena` and `!surgery` are off** until you switch them on. The names a viewer can pick are only what the game offers now: the tools on your hotbar, the arenas this game can show (no kitchen or outdoors in 2D), the surgery list |
| a round | the first valid vote opens a window (default 20 s); the option with the most votes wins if it has at least the minimum (default 2); then a cooldown (default 30 s) during which votes are ignored |
| one vote each | a later vote replaces the earlier one in the same round |
| rate limits | a viewer's messages closer than 2 s apart are ignored; the channel is capped at 30 messages a second; messages longer than 200 characters are ignored |
| what runs | the same action a player's click would: select the tool, change the arena (not saved), silence or restore a surgery group |
| on screen | a tally box (counts only, never names), the time left, and the last results |

**Privacy:** only chat lines that begin with `!` are kept, for one frame, to be counted. A viewer's name is hashed with a random salt made at start-up and only the hashes of the current round are held in memory (to stop double votes) and dropped when the round ends; nothing is written to disk.

Tests: `tests/test_streamer.py` (parsing, every rule, the connection against an in-memory fake server, a refused login, a hang-up, the guard) and `tests/test_live_inputs.py`; the playthrough's `extra:live-inputs` runs a vote against a fake server in a real game frame.
