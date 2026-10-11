//! The Twitch chat reader (`core/streamer.py:TwitchChat`) and the network switch (`core/netguard.py`).
//!
//! Anonymous and read-only: TLS to irc.chat.twitch.tv:6697, nickname `justinfan<digits>`, no password, JOIN one channel, answer PING, keep
//! only the lines that start with `!`. It never sends a chat message, never logs in, never stores a token. Only the reader thread touches its
//! socket; it notices a stop request within [`RECV_POLL`], says QUIT, closes, and takes itself off the on-screen list.

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use ktf_data::streamer::{self, Line};

pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const RECV_POLL: Duration = Duration::from_millis(250);

// --- netguard ----------------------------------------------------------------------------------------------------------------------
static DISABLED: Mutex<Option<String>> = Mutex::new(None);

/// `netguard.disable`: no network feature may open a connection from now on (headless runs call this first).
pub fn disable(reason: &str) {
    *DISABLED.lock().unwrap_or_else(PoisonError::into_inner) = Some(reason.into());
}

/// `netguard.allowed`
pub fn allowed() -> Result<(), String> {
    if let Some(r) = DISABLED.lock().unwrap_or_else(PoisonError::into_inner).as_ref() {
        return Err(format!("the network is off here ({r})"));
    }
    if std::env::var_os("KTF_NO_NETWORK").is_some_and(|v| !v.is_empty()) {
        return Err("the network is off here (KTF_NO_NETWORK is set; the test suite does this)".into());
    }
    Ok(())
}

// --- the connection ----------------------------------------------------------------------------------------------------------------
/// A byte stream the reader can use (a TLS socket, or a plain one in tests).
pub trait Wire: Read + Write + Send {
    fn set_poll(&mut self, d: Duration);
}

struct Plain(TcpStream);
impl Read for Plain {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(b)
    }
}
impl Write for Plain {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}
impl Wire for Plain {
    fn set_poll(&mut self, d: Duration) {
        let _ = self.0.set_read_timeout(Some(d));
    }
}

struct Tls(rustls::StreamOwned<rustls::ClientConnection, TcpStream>);
impl Read for Tls {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(b)
    }
}
impl Write for Tls {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.write(b)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}
impl Wire for Tls {
    fn set_poll(&mut self, d: Duration) {
        let _ = self.0.sock.set_read_timeout(Some(d));
    }
}

fn connect_tls(host: &str, port: u16) -> Result<Box<dyn Wire>, String> {
    use std::net::ToSocketAddrs;
    let addr = (host, port).to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("no address")?;
    let tcp = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).map_err(|e| e.to_string())?;
    let _ = tcp.set_read_timeout(Some(CONNECT_TIMEOUT));
    let roots = rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let cfg = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .with_root_certificates(roots)
        .with_no_client_auth();
    let name = rustls::pki_types::ServerName::try_from(host.to_string()).map_err(|e| e.to_string())?;
    let conn = rustls::ClientConnection::new(Arc::new(cfg), name).map_err(|e| e.to_string())?;
    Ok(Box::new(Tls(rustls::StreamOwned::new(conn, tcp))))
}

/// One IRC line; CR, LF and NUL are refused so nothing can add a second command.
fn line(text: &str) -> Result<Vec<u8>, String> {
    if text.contains(['\r', '\n', '\0']) {
        return Err("refused to send a line with a line break in it".into());
    }
    let mut v = text.as_bytes().to_vec();
    v.extend_from_slice(b"\r\n");
    Ok(v)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatState {
    Off,
    Connecting,
    Live,
    Error(String),
}

impl ChatState {
    pub fn name(&self) -> &'static str {
        match self {
            ChatState::Off => "off",
            ChatState::Connecting => "connecting",
            ChatState::Live => "live",
            ChatState::Error(_) => "error",
        }
    }
}

#[derive(Default)]
struct Shared {
    state: Mutex<Option<ChatState>>,
    queue: Mutex<VecDeque<(String, String)>>,
    lines_read: std::sync::atomic::AtomicU64,
    commands_read: std::sync::atomic::AtomicU64,
}

/// Reads one channel's chat. `start` opens the connection on its own thread; dropping or `stop` hangs up.
pub struct TwitchChat {
    pub channel: String,
    pub host: String,
    pub port: u16,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    factory: Arc<dyn Fn() -> Result<Box<dyn Wire>, String> + Send + Sync>,
}

impl TwitchChat {
    /// The real connection (refused when the network is off).
    pub fn new(channel: &str) -> Result<Self, String> {
        let channel = streamer::clean_channel(channel)?;
        allowed()?;
        let (host, port) = (streamer::HOST.to_string(), streamer::PORT);
        let (h, p) = (host.clone(), port);
        Ok(Self::with_factory(channel, host, port, Arc::new(move || connect_tls(&h, p))))
    }

    /// A connection made by `factory` (tests use a loopback server).
    pub fn with_factory(
        channel: String,
        host: String,
        port: u16,
        factory: Arc<dyn Fn() -> Result<Box<dyn Wire>, String> + Send + Sync>,
    ) -> Self {
        Self { channel, host, port, shared: Arc::default(), stop: Arc::new(AtomicBool::new(false)), thread: None, factory }
    }

    /// A plain (non-TLS) connection to `host:port`, for tests.
    pub fn plain_factory(host: String, port: u16) -> Arc<dyn Fn() -> Result<Box<dyn Wire>, String> + Send + Sync> {
        Arc::new(move || {
            let t = TcpStream::connect((host.as_str(), port)).map_err(|e| e.to_string())?;
            Ok(Box::new(Plain(t)) as Box<dyn Wire>)
        })
    }

    pub fn state(&self) -> ChatState {
        self.shared.state.lock().unwrap_or_else(PoisonError::into_inner).clone().unwrap_or(ChatState::Off)
    }

    pub fn commands_read(&self) -> u64 {
        self.shared.commands_read.load(Ordering::Relaxed)
    }

    pub fn lines_read(&self) -> u64 {
        self.shared.lines_read.load(Ordering::Relaxed)
    }

    pub fn describe(&self) -> String {
        format!("{}:{} #{} (read-only, anonymous)", self.host, self.port, self.channel)
    }

    pub fn start(&mut self, nick_digits: u32) {
        if matches!(self.state(), ChatState::Connecting | ChatState::Live) {
            return;
        }
        let stop = Arc::new(AtomicBool::new(false));
        self.stop = stop.clone();
        *self.shared.state.lock().unwrap_or_else(PoisonError::into_inner) = Some(ChatState::Connecting);
        let (shared, factory, channel) = (self.shared.clone(), self.factory.clone(), self.channel.clone());
        let nick = format!("justinfan{nick_digits}");
        self.thread = std::thread::Builder::new()
            .name("streamer-chat".into())
            .spawn(move || {
                let r = run(&shared, &*factory, &channel, &nick, &stop);
                if let Err(e) = r
                    && !stop.load(Ordering::Relaxed)
                {
                    *shared.state.lock().unwrap_or_else(PoisonError::into_inner) = Some(ChatState::Error(e));
                }
            })
            .ok();
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        *self.shared.state.lock().unwrap_or_else(PoisonError::into_inner) = Some(ChatState::Off);
    }

    /// The commands read since the last call: `(viewer, text)` for messages that start with `!`.
    pub fn drain(&self) -> Vec<(String, String)> {
        self.shared.queue.lock().unwrap_or_else(PoisonError::into_inner).drain(..).collect()
    }
}

impl Drop for TwitchChat {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn run(
    shared: &Shared,
    factory: &(dyn Fn() -> Result<Box<dyn Wire>, String> + Send + Sync),
    channel: &str,
    nick: &str,
    stop: &AtomicBool,
) -> Result<(), String> {
    let mut sock = factory().map_err(|e| format!("couldn't reach Twitch chat: {e}"))?;
    if stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    sock.set_poll(RECV_POLL);
    let send = |s: &mut Box<dyn Wire>, text: &str| -> Result<(), String> {
        s.write_all(&line(text)?).and_then(|()| s.flush()).map_err(|e| format!("couldn't reach Twitch chat: {e}"))
    };
    send(&mut sock, &format!("NICK {nick}"))?; // anonymous: no PASS, no token, nothing to store
    send(&mut sock, &format!("USER {nick} 0 * :{nick}"))?;
    let (mut buf, mut joined, t0) = (Vec::<u8>::new(), false, Instant::now());
    let mut chunk = [0u8; 4096];
    while !stop.load(Ordering::Relaxed) {
        match sock.read(&mut chunk) {
            Ok(0) => return Err("the connection was closed by the server".into()),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {
                if !joined && t0.elapsed() > CONNECT_TIMEOUT {
                    return Err("Twitch did not answer (no welcome within 10 seconds)".into());
                }
                continue;
            }
            Err(e) => return Err(format!("couldn't reach Twitch chat: {e}")),
        }
        if buf.len() > 65536 {
            return Err("the server sent a line that is far too long".into());
        }
        while let Some(i) = buf.iter().position(|&b| b == b'\n') {
            let raw: Vec<u8> = buf.drain(..=i).collect();
            let Some(ev) = streamer::parse_line(&String::from_utf8_lossy(&raw)) else { continue };
            shared.lines_read.fetch_add(1, Ordering::Relaxed);
            match ev {
                Line::Ping(p) => {
                    let p: String = p.chars().filter(|c| !matches!(c, '\r' | '\n' | '\0')).collect();
                    send(&mut sock, &if p.is_empty() { "PONG".to_string() } else { format!("PONG :{p}") })?;
                }
                Line::Numeric(code, _) if code == "001" && !joined => {
                    joined = true;
                    send(&mut sock, &format!("JOIN #{channel}"))?;
                    if !stop.load(Ordering::Relaxed) {
                        *shared.state.lock().unwrap_or_else(PoisonError::into_inner) = Some(ChatState::Live);
                    }
                }
                Line::Numeric(code, _) if code == "464" || code == "465" => {
                    return Err("Twitch refused the anonymous login (it may no longer allow it)".into());
                }
                Line::Notice(t)
                    if t.to_lowercase().contains("authentication failed") || t.to_lowercase().contains("improperly formatted") =>
                {
                    return Err("Twitch refused the anonymous login (it may no longer allow it)".into());
                }
                Line::Msg { user, channel: ch, text }
                    if text.starts_with('!') && text.chars().count() <= streamer::MAX_MESSAGE && ch.eq_ignore_ascii_case(channel) =>
                {
                    let mut q = shared.queue.lock().unwrap_or_else(PoisonError::into_inner);
                    if q.len() >= 200 {
                        q.pop_front();
                    }
                    q.push_back((user, text));
                    shared.commands_read.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
        }
    }
    let _ = sock.write_all(b"QUIT\r\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    use super::*;

    fn wait(chat: &TwitchChat, want: &str) {
        let t0 = Instant::now();
        while chat.state().name() != want && t0.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(chat.state().name(), want, "chat state");
    }

    /// A loopback server speaks the anonymous login, a PING, chat lines for two channels and a refusal; the reader joins, answers, keeps only
    /// `!` lines of its channel and says QUIT.
    #[test]
    fn reads_commands_answers_ping_and_quits() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (s, _) = l.accept().unwrap();
            let mut w = s.try_clone().unwrap();
            let mut r = BufReader::new(s);
            let mut got = Vec::new();
            let mut t = String::new();
            for _ in 0..2 {
                t.clear();
                r.read_line(&mut t).unwrap();
                got.push(t.trim().to_string());
            }
            w.write_all(b":tmi 001 justinfan42 :Welcome\r\n").unwrap();
            t.clear();
            r.read_line(&mut t).unwrap();
            got.push(t.trim().to_string()); // JOIN
            w.write_all(b"PING :tmi.twitch.tv\r\n").unwrap();
            t.clear();
            r.read_line(&mut t).unwrap();
            got.push(t.trim().to_string()); // PONG
            w.write_all(b":a!a@a PRIVMSG #chan :!tool swatter\r\n:b!b@b PRIVMSG #chan :hello\r\n:c!c@c PRIVMSG #other :!tool bomb\r\n")
                .unwrap();
            t.clear();
            // the QUIT arrives once the client stops
            let _ = r.read_line(&mut t);
            got.push(t.trim().to_string());
            got
        });
        let mut chat =
            TwitchChat::with_factory("chan".into(), "127.0.0.1".into(), port, TwitchChat::plain_factory("127.0.0.1".into(), port));
        chat.start(42);
        wait(&chat, "live");
        let t0 = Instant::now();
        let mut cmds = Vec::new();
        while cmds.is_empty() && t0.elapsed() < Duration::from_secs(5) {
            cmds = chat.drain();
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(cmds, vec![("a".to_string(), "!tool swatter".to_string())]);
        chat.stop();
        let got = server.join().unwrap();
        assert_eq!(got[0], "NICK justinfan42");
        assert_eq!(got[1], "USER justinfan42 0 * :justinfan42");
        assert_eq!(got[2], "JOIN #chan");
        assert_eq!(got[3], "PONG :tmi.twitch.tv");
        assert_eq!(got[4], "QUIT");
    }

    #[test]
    fn a_refusal_and_a_line_break_are_errors() {
        assert!(line("PONG :x\r\nJOIN #y").is_err());
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            s.write_all(b":tmi NOTICE * :Login authentication failed\r\n").unwrap();
            std::thread::sleep(Duration::from_millis(500));
        });
        let mut chat =
            TwitchChat::with_factory("chan".into(), "127.0.0.1".into(), port, TwitchChat::plain_factory("127.0.0.1".into(), port));
        chat.start(1);
        wait(&chat, "error");
        assert!(matches!(chat.state(), ChatState::Error(m) if m.contains("refused the anonymous login")));
        server.join().unwrap();
    }

    #[test]
    fn the_network_switch() {
        // `new` refuses when the guard says so; KTF_NO_NETWORK is how the test suite turns it off
        // SAFETY: the test sets and clears one variable no other test in this crate reads
        unsafe { std::env::set_var("KTF_NO_NETWORK", "1") };
        assert!(allowed().is_err());
        assert!(TwitchChat::new("chan").is_err());
        unsafe { std::env::remove_var("KTF_NO_NETWORK") };
        assert!(TwitchChat::new("bad name").is_err());
    }
}
