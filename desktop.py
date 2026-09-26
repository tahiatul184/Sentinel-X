"""Small local launcher; the dashboard runs in the user's browser."""
import os
import subprocess
import sys
from pathlib import Path
import threading
import time
import tkinter as tk
import webbrowser
import start_dashboard as runtime
from collection_store import Store,load_config,local_path

root=tk.Tk();root.title('AEROSENTINEL v2.9.2 | Control Center');root.geometry('780x480');root.configure(bg='#0b1820')
store=Store(runtime.APP);stop=threading.Event();message=tk.StringVar(value='Starting fully automated resilience services…')
tk.Label(root,text='AEROSENTINEL v2.9.2',font=('Segoe UI',30,'bold'),fg='#68d6c0',bg='#0b1820').pack(pady=(35,10))
tk.Label(root,text='Multimodal spatio-temporal aviation safety intelligence research',font=('Segoe UI',13),fg='white',bg='#0b1820').pack()
tk.Label(root,textvariable=message,wraplength=690,font=('Segoe UI',12),fg='#a9bdc7',bg='#0b1820').pack(pady=25)

def folder(path):
    path.mkdir(parents=True,exist_ok=True)
    if os.name=='nt':os.startfile(str(path))
    else:webbrowser.open(path.as_uri())

def toggle():store.setting('paused',not bool(store.setting('paused')))

def close():
    message.set('Stopping local services…');stop.set()
    root.after(500,wait_exit)

def wait_exit():
    if thread.is_alive():root.after(500,wait_exit)
    else:root.destroy()

def start():
    try:runtime.run(stop_event=stop,install_signals=False)
    except Exception as exc:errors.append(str(exc))

panel=tk.Frame(root,bg='#0b1820');panel.pack()
buttons=[('Open dashboard',lambda:webbrowser.open('http://127.0.0.1:8501')),
('Open satellite image archive',lambda:folder(local_path(runtime.APP,load_config(runtime.APP)['archive_root']))),
('Open raw/local image inbox',lambda:folder(local_path(runtime.APP,load_config(runtime.APP)['watch_dir']))),
('Open high-res INPUT folder',lambda:folder(runtime.APP/load_config(runtime.APP).get('geo_x_highres_inbox','highres_inbox'))),
('Open high-res processing log',lambda:folder(runtime.APP/'data/highres')),
('Pause / resume collection',toggle),('Open auto reports',lambda:folder(runtime.APP/'data/reports/auto')),
('Open service logs',lambda:folder(runtime.APP/'data/collection/logs')),('Quick start',lambda:os.startfile(str(runtime.PROJECT/'START_HERE.txt')) if os.name=='nt' else None),('Stop & Exit',close)]
for i,(label,command) in enumerate(buttons):
    tk.Button(panel,text=label,command=command,width=27,height=2,bg='#183944',fg='white',font=('Segoe UI',11),relief='flat').grid(row=i//2,column=i%2,padx=8,pady=6)
errors=[]
thread=threading.Thread(target=start,daemon=True);thread.start()
def refresh():
    if not stop.is_set():
        state=store.setting('supervisor') or {}
        if errors:message.set(errors[-1])
        elif state.get('dashboard_ready'):
            cfg=load_config(runtime.APP)
            if not cfg.get('auto_location_ready',False):
                message.set('Dashboard ready · LOCATION REQUIRED before automatic satellite collection can start')
            else:
                message.set('Dashboard ready · '+('Collection paused' if store.setting('paused') else 'Satellite archive/logging active'))
        root.after(1500,refresh)
root.after(1500,refresh);root.protocol('WM_DELETE_WINDOW',close);root.mainloop()
