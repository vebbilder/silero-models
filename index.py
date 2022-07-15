# pip install torch==1.9
import torch
from yoficator import yo
from scipy.io.wavfile import write
import numpy, re
from pygame import mixer
mixer.init(16000)
import wave
import time, os
# https://github.com/Desklop/StressRNN
from stressrnn import StressRNN
from normalizer import Normalizer
from transliterate import translit, get_available_language_codes
import numtext
import threading

# Отдельная функция-заготовка для вынесения 
# последующих функций в отдельный поток
def thread(my_func):
    def wrapper(*args, **kwargs):
        my_thread = threading.Thread(target=my_func, args=args, kwargs=kwargs)
        my_thread.start()
    return wrapper

norm = Normalizer()
stress_rnn = StressRNN()
os.environ['TORCH_HOME'] = 'silero'


def transl(text):
    text=text.lower()
    s_rus=translit(text, 'ru')
    s_rus=s_rus.replace('w','в')
    return(s_rus)

def float_to_int(wav):
  wav *= 32767
  wav = wav.astype('int16')
  return wav

language = 'ru'
speaker = 'kseniya_16khz'
torch.set_num_threads(6)
device = torch.device('cpu')
model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                      model='silero_tts',
                                                                      language=language,
                                                                      speaker=speaker)
model = model.to(device)  # gpu or cpu

def createwav(text, wavname):
    audio = apply_tts(texts=[text],
                  model=model,
                  sample_rate=sample_rate,
                  symbols=symbols,
                  device=device)
    for i, _audio in enumerate(audio):
        write(wavname, rate=16000, data=float_to_int(_audio.numpy()))

def delstress(text):
    text=text.lower()
    m=['б','й','в','г','д','ж','з','к','л','м','н','п','р','с','т','ф','х','ц','ч','ш','щ','ъ','ь']
    for xx in m:
        text=text.replace('+'+xx, xx)
    mas=text.strip().split(' ')
    if(mas[-1].count('+') > 1 and text[-2]=='+'):
        text=text[:-2]+text[-1]
        
    gl=['а','у','е','ы','а','о','э','я','и','ю','ё']
    mas=text.strip().split(' ')
    text2=''
    for z in mas:
        flag=0
        for r in z:
            if r in gl:
                flag=flag+1     
        if(flag==1 and not '+' in z):
            for r in gl:
                z=z.replace(r,'+'+r)
        text2=text2+z+' '
    text=text2+'.'
    return text
    

def sayit(text):
    stressed_text = norm.norm_text(text)
    stressed_text = stress_rnn.put_stress(text, stress_symbol='+', accuracy_threshold=0.75, replace_similar_symbols=True)
    mas=stressed_text.split('+')
    stressed_text2=''
    for x in mas:
      try:
          if(x!=''):
              s=x[:-1]+'+'+x[-1]
          else:
              s=x
          stressed_text2=stressed_text2+s
      except:
          pass
    stressed_text2=(delstress(stressed_text2))
    createwav(stressed_text2+'.', 'static/wav/'+re.sub('[^А-Яа-я]', '', text)+'.wav')


    
def playit(text):
    if(os.path.exists('static/wav/'+re.sub('[^А-Яа-я]', '', text)+'.wav')):
        s=mixer.Sound('static/wav/'+re.sub('[^А-Яа-я]', '', text)+'.wav')
        sek=s.get_length()
        s.play()
        time.sleep(sek)
        time.sleep(0.1)
    
def getpr(t):
    t=t.strip()
    t=t.replace('\n','. ')
    t=t.replace('\t','')
    t=t.replace('\r','. ')
    t=t.replace(';','. ')
    t=t.replace(',', '. ')
    t=t.replace(':', '. ')
    t=t.replace(' - ', '. ')
    t=t.replace('..','.')
    # Делим текст на массив предложений
    mas=re.split("\\b[.!?\\n]+(?=\\s)", t)
    return mas

@thread
def saybigtext(text):
    text=numtext.getnumbers(text)
    text=transl(text)
    text=yo(text)
    mas=getpr(text)
    for x in mas:
        if(len(x.strip())>1):
            sayit(x)
    return ''

def saybigtext2(text):
    text=numtext.getnumbers(text)
    text=transl(text)
    text=yo(text)
    mas=getpr(text)
    for x in mas:
        if(len(x.strip())>1):
            print(x)
            playit(x)
    return ''

def saybook(filename):
    f=open(filename, 'r', encoding='UTF-8')
    x=f.read()
    saybigtext(x)
    time.sleep(2)
    saybigtext2(x)

saybook('book.txt')

mixer.quit






