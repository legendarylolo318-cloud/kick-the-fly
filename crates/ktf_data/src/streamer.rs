//! Streamer mode's rules (`core/streamer.py`): reading one Twitch chat line, counting votes, running rounds. Every rule is a GAME RULE and
//! nothing here touches the network: the connection is `game::twitch`. Time is passed in (seconds on any monotonic clock) so every rule is
//! testable; the viewer-name hash uses a salt the caller makes at start-up.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

pub const HOST: &str = "irc.chat.twitch.tv";
pub const PORT: u16 = 6697;
pub const COMMANDS: [&str; 3] = ["tool", "surgery", "arena"];
pub const WINDOW_S: f64 = 20.0;
pub const COOLDOWN_S: f64 = 30.0;
pub const MIN_VOTES: usize = 2;
pub const USER_MIN_INTERVAL_S: f64 = 2.0;
pub const MAX_PER_SECOND: u32 = 30;
pub const MAX_MESSAGE: usize = 200;

/// A channel login: 3-25 letters, digits or underscores (a leading # is dropped). Anything else is refused so nothing a user types can add a
/// second IRC command.
pub fn clean_channel(name: &str) -> Result<String, String> {
    let c = name.trim().to_lowercase();
    let c = c.trim_start_matches('#').to_string();
    let ok = (3..=25).contains(&c.chars().count()) && c.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_');
    if ok { Ok(c) } else { Err("a channel name is 3 to 25 letters, digits or underscores (for example  mychannel)".into()) }
}

/// `slug`: lower case, runs of anything but a-z 0-9 become one `_`, trimmed.
pub fn slug(label: &str) -> String {
    let mut out = String::new();
    let mut gap = false;
    for ch in label.to_lowercase().chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() {
            if gap && !out.is_empty() {
                out.push('_');
            }
            gap = false;
            out.push(ch);
        } else {
            gap = true;
        }
    }
    out
}

/// What a line of IRC is.
#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    Ping(String),
    Msg { user: String, channel: String, text: String },
    Numeric(String, String),
    Notice(String),
}

/// `parse_line`
pub fn parse_line(line: &str) -> Option<Line> {
    let mut line = line.trim_end_matches(['\r', '\n']);
    if line.is_empty() {
        return None;
    }
    if line.starts_with('@') {
        let sp = line.find(' ')?;
        line = &line[sp + 1..];
    }
    if let Some(rest) = line.strip_prefix("PING") {
        return Some(Line::Ping(rest.strip_prefix(' ').map_or(String::new(), |r| r.trim_start_matches(':').to_string())));
    }
    let parts: Vec<&str> = line.splitn(4, ' ').collect();
    if parts.len() >= 4 && parts[1] == "PRIVMSG" && parts[0].starts_with(':') {
        let user = parts[0][1..].split('!').next().unwrap_or("").to_string();
        let text = parts[3].strip_prefix(':').unwrap_or(parts[3]).to_string();
        return Some(Line::Msg { user, channel: parts[2].trim_start_matches('#').to_string(), text });
    }
    if parts.len() >= 2 && !parts[1].is_empty() && parts[1].chars().all(|c| c.is_ascii_digit()) {
        return Some(Line::Numeric(parts[1].to_string(), line.to_string()));
    }
    if parts.len() >= 2 && parts[1] == "NOTICE" {
        return Some(Line::Notice(line.to_string()));
    }
    None
}

/// `parse_command`: `("tool", "frog")` for `!tool frog`.
pub fn parse_command(text: &str) -> Option<(String, String)> {
    let t = text.trim();
    if !t.starts_with('!') || t.chars().count() > MAX_MESSAGE {
        return None;
    }
    let bits: Vec<&str> = t[1..].split_whitespace().collect();
    if bits.len() < 2 || !COMMANDS.contains(&bits[0].to_lowercase().as_str()) {
        return None;
    }
    Some((bits[0].to_lowercase(), slug(&bits[1..].join(" "))))
}

#[derive(Debug, Clone)]
pub struct Rules {
    pub window_s: f64,
    pub cooldown_s: f64,
    pub min_votes: usize,
    pub allow: Vec<String>,
}

impl Default for Rules {
    fn default() -> Self {
        Self { window_s: WINDOW_S, cooldown_s: COOLDOWN_S, min_votes: MIN_VOTES, allow: vec!["tool".into()] }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub command: String,
    pub slug: String,
    pub label: String,
    pub votes: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Idle,
    Open,
    Cooldown,
}

impl State {
    pub fn name(self) -> &'static str {
        match self {
            State::Idle => "idle",
            State::Open => "open",
            State::Cooldown => "cooldown",
        }
    }
}

type Hash = [u8; 12];

/// `VoteBoard`: counts votes and runs the rounds.
pub struct VoteBoard {
    pub rules: Rules,
    /// command -> slug -> label
    pub options: HashMap<String, Vec<(String, String)>>,
    salt: [u8; 16],
    pub state: State,
    round_end: f64,
    cool_end: f64,
    votes: HashMap<Hash, (String, String, f64)>,
    last_seen: HashMap<Hash, f64>,
    sec: i64,
    sec_n: u32,
    pub ignored: u64,
}

impl VoteBoard {
    pub fn new(salt: [u8; 16]) -> Self {
        Self {
            rules: Rules::default(),
            options: HashMap::new(),
            salt,
            state: State::Idle,
            round_end: 0.0,
            cool_end: 0.0,
            votes: HashMap::new(),
            last_seen: HashMap::new(),
            sec: 0,
            sec_n: 0,
            ignored: 0,
        }
    }

    /// Votes already cast for a command no longer allowed are dropped (a command switched off mid-round can't win it).
    pub fn set_rules(&mut self, rules: Rules) {
        self.votes.retain(|_, (c, _, _)| rules.allow.contains(c));
        self.rules = rules;
    }

    fn hash(&self, user: &str) -> Hash {
        let mut h = Sha256::new();
        h.update(self.salt);
        h.update(user.to_lowercase().as_bytes());
        let d = h.finalize();
        let mut out = [0u8; 12];
        out.copy_from_slice(&d[..12]);
        out
    }

    fn label_of(&self, command: &str, slug: &str) -> Option<&str> {
        self.options.get(command)?.iter().find(|(s, _)| s == slug).map(|(_, l)| l.as_str())
    }

    /// Count one chat message; says why it did or did not count.
    pub fn submit(&mut self, user: &str, text: &str, now: f64) -> &'static str {
        let sec = now.floor() as i64;
        if sec != self.sec {
            self.sec = sec;
            self.sec_n = 0;
        }
        self.sec_n += 1;
        if self.sec_n > MAX_PER_SECOND {
            self.ignored += 1;
            return "channel rate limit";
        }
        let Some((command, arg)) = parse_command(text) else { return "not a command" };
        if !self.rules.allow.contains(&command) {
            self.ignored += 1;
            return "command not allowed";
        }
        if self.label_of(&command, &arg).is_none() {
            self.ignored += 1;
            return "not an option";
        }
        if self.state == State::Cooldown {
            if now < self.cool_end {
                self.ignored += 1;
                return "cooldown";
            }
            self.state = State::Idle;
        }
        let h = self.hash(user);
        if let Some(&last) = self.last_seen.get(&h)
            && now - last < USER_MIN_INTERVAL_S
        {
            self.ignored += 1;
            return "too fast";
        }
        self.last_seen.insert(h, now);
        if self.last_seen.len() > 5000 {
            self.last_seen.retain(|_, v| now - *v < 60.0);
        }
        if self.state == State::Idle {
            self.state = State::Open;
            self.round_end = now + self.rules.window_s;
            self.votes.clear();
        }
        self.votes.insert(h, (command, arg, now));
        "counted"
    }

    /// `(command, slug, label, votes)`, most votes first (ties: the earliest first vote). No names.
    pub fn tally(&self) -> Vec<(String, String, String, usize)> {
        let mut counts: HashMap<(String, String), (usize, f64)> = HashMap::new();
        for (c, a, when) in self.votes.values() {
            let e = counts.entry((c.clone(), a.clone())).or_insert((0, f64::INFINITY));
            e.0 += 1;
            e.1 = e.1.min(*when);
        }
        let mut rows: Vec<((String, String), (usize, f64))> = counts.into_iter().collect();
        // Python sorts stably over insertion order; the key (-votes, first vote time) decides everything but exact ties of both
        rows.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(a.1.1.total_cmp(&b.1.1)).then(a.0.cmp(&b.0)));
        rows.into_iter()
            .map(|((c, a), (n, _))| {
                let label = self.label_of(&c, &a).map_or_else(|| a.clone(), String::from);
                (c, a, label, n)
            })
            .collect()
    }

    /// Close the round when its window is over; the winner once, or `None`.
    pub fn tick(&mut self, now: f64) -> Option<Outcome> {
        if self.state == State::Cooldown && now >= self.cool_end {
            self.state = State::Idle;
        }
        if self.state != State::Open || now < self.round_end {
            return None;
        }
        let rows = self.tally();
        let total = self.votes.len();
        self.votes.clear();
        self.last_seen.retain(|_, v| now - *v < USER_MIN_INTERVAL_S);
        self.state = State::Cooldown;
        self.cool_end = now + self.rules.cooldown_s;
        let (c, a, label, n) = rows.into_iter().next()?;
        (n >= self.rules.min_votes).then_some(Outcome { command: c, slug: a, label, votes: n, total })
    }

    pub fn remaining(&self, now: f64) -> f64 {
        match self.state {
            State::Open => (self.round_end - now).max(0.0),
            State::Cooldown => (self.cool_end - now).max(0.0),
            State::Idle => 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.votes.clear();
        self.last_seen.clear();
        self.state = State::Idle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board() -> VoteBoard {
        let mut b = VoteBoard::new([7; 16]);
        b.options.insert("tool".into(), vec![("swatter".into(), "SWATTER".into()), ("bomb".into(), "BOMB".into())]);
        b.options.insert("arena".into(), vec![("pool".into(), "pool".into())]);
        b
    }

    #[test]
    fn channels_slugs_and_lines() {
        assert_eq!(clean_channel(" #MyChannel ").unwrap(), "mychannel");
        assert!(clean_channel("a b").is_err() && clean_channel("ab").is_err() && clean_channel("x\r\nJOIN #y").is_err());
        assert_eq!(slug("  Hello, World!! "), "hello_world");
        assert_eq!(parse_command("!Tool  Frog  Food"), Some(("tool".into(), "frog_food".into())));
        assert_eq!(parse_command("!tool"), None);
        assert_eq!(parse_command("hello"), None);
        assert_eq!(
            parse_line("@badge=1;x=y :bob!bob@bob.tmi.twitch.tv PRIVMSG #chan :!tool swatter"),
            Some(Line::Msg { user: "bob".into(), channel: "chan".into(), text: "!tool swatter".into() })
        );
        assert_eq!(parse_line("PING :tmi.twitch.tv"), Some(Line::Ping("tmi.twitch.tv".into())));
        assert_eq!(parse_line(":tmi 001 justinfan1 :Welcome"), Some(Line::Numeric("001".into(), ":tmi 001 justinfan1 :Welcome".into())));
    }

    #[test]
    fn rounds_cooldown_and_limits() {
        let mut b = board();
        assert_eq!(b.submit("a", "!tool swatter", 0.0), "counted");
        assert_eq!(b.submit("a", "!tool bomb", 0.5), "too fast");
        assert_eq!(b.submit("b", "!tool swatter", 1.0), "counted");
        assert_eq!(b.submit("c", "!tool bomb", 1.5), "counted");
        assert_eq!(b.submit("d", "!arena pool", 1.6), "command not allowed");
        assert_eq!(b.submit("e", "!tool nothing", 1.7), "not an option");
        assert!(b.tick(19.9).is_none());
        let r = b.tick(20.0).expect("two votes win");
        assert_eq!((r.slug.as_str(), r.votes, r.total), ("swatter", 2, 3));
        assert_eq!(b.state, State::Cooldown);
        assert_eq!(b.submit("f", "!tool bomb", 25.0), "cooldown");
        assert_eq!(b.submit("f", "!tool bomb", 51.0), "counted");
        // a round below the minimum closes with no winner
        assert!(b.tick(72.0).is_none());
        assert_eq!(b.state, State::Cooldown);
    }

    #[test]
    fn a_command_switched_off_cannot_win() {
        let mut b = board();
        b.set_rules(Rules { allow: vec!["tool".into(), "arena".into()], ..Rules::default() });
        b.submit("a", "!arena pool", 0.0);
        b.submit("b", "!arena pool", 1.0);
        b.set_rules(Rules::default());
        assert!(b.tick(21.0).is_none());
    }
}
