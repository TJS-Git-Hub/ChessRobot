"""Synchronous UCI protocol communication. Run search in worker thread from UI."""
import os, subprocess, threading, queue, time

class EngineError(RuntimeError):pass

class UCIEngine:
    def __init__(self, exe, threads=2, hash_mb=128):
        if not os.path.isfile(exe):raise EngineError('未找到引擎文件: '+exe)
        self.proc=subprocess.Popen([os.path.abspath(exe)],cwd=os.path.dirname(os.path.abspath(exe)),
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1,
            encoding='utf-8',errors='replace')
        self.q=queue.Queue(); self.lock=threading.Lock()
        self.searching=False
        self.reader = threading.Thread(target=self._reader,daemon=True)
        self.reader.start()
        try:
            with self.lock:
                self._send('uci'); self._until('uciok',10)
                self._send(f'setoption name Threads value {threads}')
                self._send(f'setoption name Hash value {hash_mb}')
                self._send('isready');self._until('readyok',15)
        except Exception:
            self.close()
            raise
    def _reader(self):
        for line in self.proc.stdout:self.q.put(line.strip())
    def _send(self,command):
        if self.proc.poll() is not None:raise EngineError('引擎进程已经退出，检查 NNUE 文件是否与可执行文件在同一目录')
        self.proc.stdin.write(command+'\n');self.proc.stdin.flush()
    def _until(self,target,timeout):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.proc.poll() is not None and self.q.empty():
                raise EngineError('引擎进程已经退出，检查 NNUE 权重及 CPU 兼容性')
            try:line=self.q.get(timeout=min(.25,max(.01,deadline-time.monotonic())))
            except queue.Empty:continue
            if line.startswith('info string CRITICAL ERROR'):raise EngineError(line)
            if line.startswith(target):return line
        raise EngineError('引擎等待超时: '+target)
    def bestmove(self,fen,movetime_ms=1500):
        with self.lock:
            self._send('position fen '+fen)
            self.searching=True
            try:
                self._send('go movetime '+str(max(100,int(movetime_ms))))
                line=self._until('bestmove',max(15,movetime_ms/1000+15))
            finally:
                self.searching=False
            return line.split()[1] if len(line.split())>1 else None
    def stop(self):
        """Request the running search to stop early (UCI 'stop'). Safe to call from UI thread."""
        try:
            if self.searching:
                self.proc.stdin.write('stop\n'); self.proc.stdin.flush()
        except Exception:
            pass
    def close(self):
        try:self._send('quit')
        except Exception:pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2)
        self.reader.join(timeout=2)
        self.proc.stdin.close()
        self.proc.stdout.close()
