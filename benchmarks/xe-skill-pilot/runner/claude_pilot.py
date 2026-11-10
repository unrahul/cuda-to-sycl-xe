#!/usr/bin/env python3
"""Bash-only Claude pilot; trusted mailbox invokes a fixed assessment controller."""
import argparse, json, os, pathlib, re, shutil, signal, stat, subprocess, tempfile, threading, time

p=argparse.ArgumentParser()
p.add_argument('--workdir', required=True, type=pathlib.Path)
p.add_argument('--prompt', required=True, type=pathlib.Path)
p.add_argument('--result', required=True, type=pathlib.Path)
p.add_argument('--task-id', required=True)
p.add_argument('--read-only', action='append', default=[], help='workspace-relative task or frozen packet path')
p.add_argument('--controller', required=True, type=pathlib.Path)
p.add_argument('--run-dir', required=True, type=pathlib.Path)
p.add_argument('--probe-only', action='store_true')
p.add_argument('--broker-self-test', action='store_true')
a=p.parse_args()
work=a.workdir.resolve(strict=True); result=a.result.resolve(); controller=a.controller.resolve(strict=True); run_dir=a.run_dir.resolve()
if not re.fullmatch(r'L[12]_[A-Za-z0-9_]+',a.task_id) and not (a.task_id=='PREFLIGHT_ONLY' and controller.name=='fixture_controller.py'): p.error('invalid task ID')
if not work.is_dir() or not work.is_relative_to('/tmp') or work == pathlib.Path('/tmp'): p.error('workdir must be a dedicated directory beneath /tmp')
if result == work or work in result.parents or run_dir == work or work in run_dir.parents: p.error('results and run-dir must be outside workdir')
if result.exists(): p.error('result already exists')
if any((x/'CLAUDE.md').exists() or (x/'AGENTS.md').exists() for x in work.parents if x != pathlib.Path('/')): p.error('ancestor instructions found')
if not controller.is_file(): p.error('missing controller')
readonly=[]
for name in a.read_only:
 rel=pathlib.PurePosixPath(name)
 if rel.is_absolute() or not rel.parts or any(part in ('.','..') for part in rel.parts): p.error('invalid read-only path')
 path=work.joinpath(*rel.parts)
 if not path.exists() or not path.resolve().is_relative_to(work) or path.is_symlink(): p.error('invalid read-only target')
 readonly.append(str(rel))
claude=pathlib.Path(shutil.which('claude') or '').resolve(strict=True)
codex=pathlib.Path(shutil.which('codex') or '').resolve(strict=True)
auth=pathlib.Path.home()/'.claude/.credentials.json'
if not auth.is_file(): p.error('file-backed Claude auth not available')
result.mkdir(parents=True,mode=0o700)
for name in ['requests','feedback','submission']:
 q=work/name
 if q.exists() and (q.is_symlink() or not q.is_dir() or any(q.iterdir())): p.error(name+' must be absent or empty')
 q.mkdir(mode=0o700,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='xe-claude-control-',dir='/tmp') as tmp:
 base=pathlib.Path(tmp); home=base/'home'; home.mkdir(mode=0o700); conf=home/'.claude'; conf.mkdir(mode=0o700); ch=base/'codex'; ch.mkdir(mode=0o700)
 shutil.copyfile(auth,conf/'.credentials.json'); (conf/'.credentials.json').chmod(0o600)
 original=json.loads((pathlib.Path.home()/'.claude.json').read_text())
 (home/'.claude.json').write_text(json.dumps({k:original[k] for k in ['hasCompletedOnboarding','lastOnboardingVersion','oauthAccount'] if k in original})); (home/'.claude.json').chmod(0o600)
 (ch/'config.toml').write_text('''default_permissions = "pilot"\napproval_policy = "never"\n[permissions.pilot.filesystem]\n":minimal" = "read"\n'''+json.dumps(str(codex))+' = "read"\n'+'''[permissions.pilot.filesystem.":workspace_roots"]\n"." = "write"\n'''+''.join(json.dumps(name)+' = "read"\n' for name in readonly)+'''[permissions.pilot.network]\nenabled = false\n''')
 wrapper=base/'shell-wrapper'
 wrapper.write_text('#!/usr/bin/python3\nimport os,sys\nos.environ["CODEX_HOME"]='+repr(str(ch))+'\nos.execv('+repr(str(codex))+',['+repr(str(codex))+',"sandbox","-P","pilot","-C",'+repr(str(work))+',"--","/bin/bash","-c",sys.argv[1]])\n'); wrapper.chmod(0o700)
 sentinel=base/'hidden'; sentinel.write_text('x')
 probe='''import os,socket,sys,json
c={}
for k,path in [('auth',sys.argv[1]),('control',sys.argv[2]),('repo','/home/rahul/Coding'),('docker','/var/run/docker.sock')]: c[k]=not os.path.exists(path)
try: s=socket.socket();s.connect(('1.1.1.1',443));c['internet']=False
except OSError: c['internet']=True
try: open('.probe','w').write('ok');os.unlink('.probe');c['write']=True
except OSError: c['write']=False
for path in sys.argv[3:]:
 try:
  if os.path.isdir(path):
   nested=None
   for root,dirs,files in os.walk(path,followlinks=False):
    dirs[:]=[d for d in dirs if not os.path.islink(os.path.join(root,d))]
    for name in files:
     f=os.path.join(root,name)
     if os.path.isfile(f) and not os.path.islink(f): nested=f;break
    if nested: break
   if not nested: c['nested_readonly:'+path]=False
   else:
    try: fd=os.open(nested,os.O_WRONLY|os.O_NOFOLLOW);os.close(fd);c['nested_readonly:'+path]=False
    except OSError: c['nested_readonly:'+path]=True
   test=os.path.join(path,'.xe-readonly-probe');fd=os.open(test,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd);os.unlink(test)
  else:
   fd=os.open(path,os.O_WRONLY);os.close(fd)
  c['readonly:'+path]=False
 except OSError: c['readonly:'+path]=True
print(json.dumps(c));sys.exit(0 if all(c.values()) else 1)'''
 env={'HOME':str(home),'CLAUDE_CONFIG_DIR':str(conf),'CLAUDE_CODE_SHELL_PREFIX':str(wrapper),'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC':'1','CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL':'1','CLAUDE_CODE_DISABLE_POLICY_SKILLS':'1','CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK':'1','PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C.UTF-8','TMPDIR':str(work)}
 cmd=[str(codex),'sandbox','-P','pilot','-C',str(work),'--','/usr/bin/python3','-c',probe,str(conf/'.credentials.json'),str(sentinel),*readonly]
 probe_env=dict(env,CODEX_HOME=str(ch)); r=subprocess.run(cmd,env=probe_env,capture_output=True,text=True,timeout=20)
 (result/'sandbox-probe.json').write_text(json.dumps({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2))
 if r.returncode: raise SystemExit('sandbox probe failed; model not launched')
 if a.probe_only: raise SystemExit(0)
 reqfd=os.open(work/'requests',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 feedbackfd=os.open(work/'feedback',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 submissionfd=os.open(work/'submission',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 staging=base/'staging';staging.mkdir(mode=0o700)
 def snapshot_submission(dest):
  dest.mkdir(mode=0o700)
  limits={'count':0,'bytes':0,'entries':0}
  allowed={'.py','.cpp','.cc','.cxx','.c','.hpp','.h','.sycl','.json','.txt'}
  def copydir(srcfd,dst,depth=0):
   if depth>16: raise ValueError('submission too deep')
   for name in os.listdir(srcfd):
    limits['entries']+=1
    if limits['entries']>512: raise ValueError('too many entries')
    info=os.stat(name,dir_fd=srcfd,follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode): raise ValueError('submission symlink')
    if stat.S_ISDIR(info.st_mode):
     sub=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=srcfd)
     try:
      target=dst/name;target.mkdir();copydir(sub,target,depth+1)
     finally: os.close(sub)
    elif stat.S_ISREG(info.st_mode):
     if pathlib.Path(name).suffix.lower() not in allowed: continue
     fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=srcfd)
     try:
      actual=os.fstat(fd)
      if not stat.S_ISREG(actual.st_mode) or actual.st_size>10*1024*1024: raise ValueError('bad file')
      limits['count']+=1;limits['bytes']+=actual.st_size
      if limits['count']>128 or limits['bytes']>64*1024*1024: raise ValueError('submission too large')
      with os.fdopen(os.dup(fd),'rb') as source, (dst/name).open('xb') as target:
       remaining=actual.st_size
       while remaining:
        chunk=source.read(min(1024*1024,remaining))
        if not chunk: raise ValueError('file shrank during snapshot')
        target.write(chunk);remaining-=len(chunk)
       if source.read(1): raise ValueError('file grew during snapshot')
      after=os.fstat(fd)
      if (actual.st_size,actual.st_mtime_ns,actual.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns): raise ValueError('file changed during snapshot')
     finally: os.close(fd)
    else: raise ValueError('submission nonregular entry')
  copydir(submissionfd,dest)
  candidate=dest/'model_new.py'
  if not candidate.is_file(): raise ValueError('candidate missing')
  return candidate
 stop=threading.Event();broker_fault={'value':None}
 def broker():
  processed=set()
  while not stop.wait(0.25):
   for n in range(1,7):
    if stop.is_set(): break
    if n in processed: continue
    try:
     fd=os.open(f'{n}.json',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=reqfd)
    except (OSError,ValueError): continue
    try:
     st=os.fstat(fd)
     if not stat.S_ISREG(st.st_mode) or st.st_size>1024: continue
     data=json.loads(os.read(fd,1025))
     if data != {'id':n}: continue
    except (OSError,ValueError): continue
    finally: os.close(fd)
    processed.add(n)
    cleanup_active=False
    try:
     candidate=snapshot_submission(staging/str(n))
     command=['/usr/bin/python3',str(controller),'assess','--task-id',a.task_id,'--candidate',str(candidate),'--run-dir',str(run_dir),'--phase','assessment']
     # Controller internally bounds flock to 600s and work to 1800s. Give it
     # 2700s including cleanup; on overrun terminate only its process group and
     # remove only its uniquely named Docker containers.
     assess=subprocess.Popen(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
     try:
      assess_out,assess_err=assess.communicate(timeout=2700)
      assess_code=assess.returncode
     except subprocess.TimeoutExpired:
      cleanup_active=True
      try: os.killpg(assess.pid,signal.SIGTERM)
      except ProcessLookupError: pass
      try: assess.communicate(timeout=10)
      except subprocess.TimeoutExpired:
       try: os.killpg(assess.pid,signal.SIGKILL)
       except ProcessLookupError: pass
       assess.communicate()
      names=subprocess.run(['docker','ps','-aq','--filter',f'name=^/xe-pilot-{assess.pid}-'],text=True,capture_output=True,timeout=30)
      ids=[x for x in names.stdout.splitlines() if re.fullmatch(r'[0-9a-f]{12,64}',x)]
      if names.returncode: raise RuntimeError('controller cleanup inventory failed')
      if ids:
       cleanup=subprocess.run(['docker','rm','-f',*ids],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
       if cleanup.returncode: raise RuntimeError('controller cleanup failed')
      confirm=subprocess.run(['docker','ps','-aq','--filter',f'name=^/xe-pilot-{assess.pid}-'],text=True,capture_output=True,timeout=30)
      if confirm.returncode or confirm.stdout.strip(): raise RuntimeError('controller cleanup unverified')
      raise TimeoutError('assessment controller exceeded 2700 seconds')
     try:
      raw=json.loads(assess_out)
      feedback={k:raw.get(k) for k in ('task_id','status','correct','sycl_backend','baseline_median_ms','candidate_median_ms','speedup') if isinstance(raw.get(k),(str,int,float,bool,type(None)))}
      feedback['errors']=[]
      for e in raw.get('errors',[]):
       if isinstance(e,dict) and isinstance(e.get('category'),str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}',e['category']):
        clean={'category':e['category']}
        if e['category'] in ('candidate_load','candidate_policy','candidate_build') and isinstance(e.get('message'),str): clean['message']=e['message'][:6000 if e['category']=='candidate_build' else 3000]
        feedback['errors'].append(clean)
     except (ValueError,AttributeError,TypeError): feedback={'status':'controller_error','exit':assess_code}
    except TimeoutError: feedback={'status':'controller_timeout'}
    except RuntimeError: feedback={'status':'controller_cleanup_unverified'};broker_fault['value']='controller_cleanup_unverified'
    except (OSError,ValueError,subprocess.TimeoutExpired):
     if cleanup_active: feedback={'status':'controller_cleanup_unverified'};broker_fault['value']='controller_cleanup_unverified'
     else: feedback={'status':'candidate_or_controller_error'}
    path=f'{n}.json'; temp=f'.{n}.tmp'
    # Agent can replace mailbox entries; refuse links and use exclusive creation.
    try:
     fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=feedbackfd)
     with os.fdopen(fd,'w') as f: json.dump(feedback,f)
     os.replace(temp,path,src_dir_fd=feedbackfd,dst_dir_fd=feedbackfd)
    except OSError: pass
 thread=threading.Thread(target=broker,daemon=False); thread.start()
 if a.broker_self_test:
  (work/'submission/model_new.py').write_text('fixture')
  (work/'requests/1.json').write_text('{\"id\":1}')
  deadline=time.monotonic()+20
  while time.monotonic()<deadline and not (work/'feedback/1.json').is_file(): time.sleep(0.1)
  stop.set(); thread.join(timeout=5)
  if not (work/'feedback/1.json').is_file(): raise SystemExit('broker self-test failed')
  print((work/'feedback/1.json').read_text()); raise SystemExit(0)
 prompt=a.prompt.read_text()
 settings=json.dumps({'disableClaudeAiConnectors':True,'syncClaudeAiPlugins':False})
 cmd=[str(claude),'-p','--safe-mode','--disable-slash-commands','--strict-mcp-config','--no-chrome','--no-session-persistence','--tools','Bash','--allowedTools','Bash','--permission-mode','dontAsk','--settings',settings,'--model','claude-opus-5-5','--effort','high','--output-format','stream-json','--verbose']
 (result/'invocation.json').write_text(json.dumps({'command':cmd,'cwd':str(work),'task_id':a.task_id,'model_time_limit_seconds':7200,'model_turn_limit':120,'assessment_timeout_seconds':2700},indent=2))
 try:
  with (result/'events.jsonl').open('w') as out,(result/'stderr.log').open('w') as err:
   proc=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,start_new_session=True,bufsize=1)
   proc.stdin.write(prompt);proc.stdin.close(); init=False; violation=None; budget_hit=None; message_ids=set()
   def time_limit():
    budget_state['value']='time_limit_7200_seconds'
    try:
     os.killpg(proc.pid,signal.SIGTERM)
     time.sleep(10)
     if proc.poll() is None: os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError: pass
   budget_state={'value':None}
   watchdog=threading.Timer(7200,time_limit);watchdog.start()
   for line in proc.stdout:
    out.write(line);out.flush()
    try: event=json.loads(line)
    except ValueError: violation='invalid JSON event'; os.killpg(proc.pid,signal.SIGTERM); break
    if event.get('type')=='system' and event.get('subtype')=='init':
     init=True
     if event.get('tools')!=['Bash'] or event.get('mcp_servers') or event.get('skills') or event.get('slash_commands') or pathlib.Path(event.get('cwd','/')).resolve()!=work or event.get('model')!='claude-opus-5-5':
      violation='unexpected init tool, context, cwd, or model'; os.killpg(proc.pid,signal.SIGTERM); break
    if event.get('type')=='assistant':
     message_id=event.get('message',{}).get('id')
     if message_id and message_id not in message_ids:
      message_ids.add(message_id)
      if len(message_ids)>120:
       budget_state['value']='turn_limit_120';os.killpg(proc.pid,signal.SIGTERM);break
     for item in event.get('message',{}).get('content',[]):
      if item.get('type')=='tool_use' and item.get('name')!='Bash': violation='unexpected tool call';os.killpg(proc.pid,signal.SIGTERM);break
   try: proc.wait(timeout=10)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   watchdog.cancel(); budget_hit=budget_state['value']
   if not init: violation='missing init event'
 finally:
  stop.set();thread.join()
 # Status is published only after any in-flight assessment has completed or
 # received bounded cleanup, so the parent cannot select a partial result.
 (result/'status.json').write_text(json.dumps({'exit':proc.returncode,'violation':violation,'budget_hit':budget_hit,'model_turn_ids_seen':len(message_ids),'broker_fault':broker_fault['value']},indent=2))
 raise SystemExit(1 if violation or budget_hit or broker_fault['value'] or proc.returncode else 0)
