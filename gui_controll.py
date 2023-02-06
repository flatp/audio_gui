import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import messagebox
from tkinter import filedialog
import pygame
import scipy.io.wavfile
import math
import os
import shutil
import time
# MatplotlibをTkinterで使用するために必要
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def file_select():
  idir = 'C:\\python_test' #初期フォルダ
  filetype = [("音楽","*.mp3"), ("音楽","*.wav")] #拡張子の選択
  file_path = tkinter.filedialog.askopenfilename(filetypes = filetype, initialdir = idir)
  input_box.insert(tkinter.END, file_path)
  guidayo(file_path)

def click_close():
    if messagebox.askokcancel("終了確認", "プログラムを終了しますか？"):
        quit()

def is_peak(a, index):
	if index == 0:
		return a[0] > a[1]
	elif index == len(a) - 1:
		return a[index] > a[index - 1]
	else:
		return a[index] > a[index - 1] and a[index] > a[index + 1]

def zero_cross(waveform):
	
	zc = 0

	for i in range(len(waveform) - 1):
		if(
			(waveform[i] > 0.0 and waveform[i+1] < 0.0) or
			(waveform[i] < 0.0 and waveform[i+1] > 0.0)
		):
			zc += 1
	
	return zc

def cepstrum(amplitude_spectrum):
	log_spectrum = np.log(amplitude_spectrum)
	cepstrum = np.fft.fft(log_spectrum)
	return cepstrum

def hz2nn(frequency):
	return int (round (12.0 * (np.log(frequency / 440.0) / np.log (2.0)))) + 69

def chroma_vector(spectrum, frequencies):
	
	# 0 = C, 1 = C#, 2 = D, ..., 11 = B

	# 12次元のクロマベクトルを作成（ゼロベクトルで初期化）
	cv = np.zeros(12)
	
	# スペクトルの周波数ビン毎に
	# クロマベクトルの対応する要素に振幅スペクトルを足しこむ
	for s, f in zip (spectrum , frequencies):
		nn = hz2nn(f)
		cv[nn % 12] += np.abs(s)
	
	return cv

def chroma_display(title):
	SR = 16000
	x, _ = librosa.load(title, sr=SR)

	size_frame = 2048
	hamming_window = np.hamming(size_frame)
	size_shift = 16000 / 100
	chroma = []
	waon = []

	for i in np.arange(0, len(x)-size_frame, size_shift):
		idx = int(i)
		x_frame = x[idx : idx+size_frame]

		fft_spec = np.fft.rfft(x_frame * hamming_window)
		fre = np.linspace(8000/len(np.abs(fft_spec)), 8000, len(np.abs(fft_spec)))
		cv = chroma_vector(np.abs(fft_spec), fre)
		chroma.append(cv)
		max = -10000
		flag = 0
		for j in range(0,24):
			fla = cv[j//2] * 1 + cv[(j//2+4-(j%2))%12] * 0.5 + cv[(j//2+7)%12] * 0.8
			if fla > max :
				max = fla
				flag = j

		waon.append(flag)

	root1 = tkinter.Tk()
	root1.wm_title("CHROMA_DISPLAY")
	fig_c = plt.figure()
	canvas_c = FigureCanvasTkAgg(fig_c, master=root1)
	ax21 = fig_c.add_subplot(2, 1, 1)
	ax22 = fig_c.add_subplot(2, 1, 2)
	ax21.set_ylabel('chromavector')		# y軸のラベルを設定
	ax21.imshow(
		np.flipud(np.array(chroma).T),		# 画像とみなすために，データを転置して上下反転
		extent=[0, len(x) / SR, 0, 12],			# (横軸の原点の値，横軸の最大値，縦軸の原点の値，縦軸の最大値)
		aspect='auto',
		interpolation='nearest'
	)
	xx_data = [xx * size_shift / SR for xx in range(len(waon))] 
	ax22.set_xlabel('time[s]')					# x軸のラベルを設定
	ax22.set_ylabel('chord_name')		# y軸のラベルを設定
	ax22.set_xlim(0, len(x) / SR)
	ax22.plot(xx_data, waon)
	canvas_c.get_tk_widget().pack(side="left")

def generate_sinusoid(sampling_rate, frequency, duration):
	sampling_interval = 1.0 / sampling_rate
	t = np.arange(sampling_rate * duration) * sampling_interval
	waveform = np.sin(2.0 * math.pi * frequency * t)
	return waveform
    


def guidayo(filename):
    size_frame = 512	# フレームサイズ
    size_shift = 16000 // 100	# シフトサイズ = 0.001 秒 (10 msec)

	# 音声ファイルを読み込む
    x, _ = librosa.load(filename, sr=SR)

	# ファイルサイズ（秒）
    duration = len(x) / SR

	# ハミング窓
    hamming_window = np.hamming(size_frame)

	# スペクトログラムを保存するlist
    spectrogram = []
    spectrogram4 = []
    basics = []
    sikibetu = []
    voll =[]

	# フレーム毎にスペクトルを計算
    for i in np.arange(0, len(x)-size_frame, size_shift):
		
		# 該当フレームのデータを取得
        idx = int(i)	# arangeのインデクスはfloatなのでintに変換
        x_frame = x[idx : idx+size_frame]
		
		# スペクトル
        fft_spec = np.fft.rfft(x_frame * hamming_window)
        fft_log_abs_spec = np.log(np.abs(fft_spec))
        spectrogram.append(fft_log_abs_spec)
        spectrogram4.append(fft_log_abs_spec[:len(fft_log_abs_spec)//4])
        cep = cepstrum(np.abs(fft_spec))
        cepss = [np.real(cep[ii]) for ii in range(0,13)]
        prejudge = 0 
        sikibetusi = 0

	# 自己相関が格納された，長さが len(x)*2-1 の対称な配列を得る
        autocorr = np.correlate(x_frame * hamming_window, x_frame * hamming_window, 'full')

	# 不要な前半を捨てる
        autocorr = autocorr [len (autocorr ) // 2 : ]

	# ピークのインデックスを抽出する
        peakindices = [i for i in range (len (autocorr )) if is_peak (autocorr, i)]

	# インデックス0 がピークに含まれていれば捨てる
        peakindices = [i for i in peakindices if i != 0]

	# 自己相関が最大となるインデックスを得る
        rms = np.sqrt(np.sum(np.square(np.abs(x_frame))) / len(x_frame))
        vol = 20 * np.log10(rms)
        voll.append(vol)

        zero = zero_cross(x_frame)

        try:
            max_peak_index = max (peakindices , key=lambda index: autocorr [index])
            if 16000 / max_peak_index < 300 and zero < 100 and vol > -30:
                basics.append(16000 / max_peak_index)
            else:
                basics.append(0)
        except:
            basics.append(0)

        for j in range(0,5):
            bunbo = 1
            for k in range(0,13):
                bunbo = bunbo*sigma[j][k]
            judge = np.exp(- sum([(cepss[ii] - myu[j][ii])**2 / (2*sigma[j][ii]) for ii in range(0,13)])) / np.sqrt(bunbo)
            if judge > prejudge:
                sikibetusi = j
                prejudge = judge 

        if vol < -35:
            sikibetusi = -1
        sikibetu.append(sikibetusi)

	# Tkinterを初期化
    root = tkinter.Tk()
    root.wm_title("EXP4-AUDIO-SAMPLE")
	# root.geometry("500x500")

	# Tkinterのウィジェットを階層的に管理するためにFrameを使用
	# frame1 ... スペクトログラムを表示
	# frame2 ... Scale（スライドバー）とスペクトルを表示
    frame1 = tkinter.Frame(root, width=200, height=200)
    frame2 = tkinter.Frame(root, width=200, height=200)
    frame3 = tkinter.Frame(root, width=200, height=40)
    frame4 = tkinter.Frame(root, width=400, height=20)
    frame5 = tkinter.Frame(root, width=400, height=20)
    frame6 = tkinter.Frame(root, width=400, height=20)
    frame7 = tkinter.Frame(root, width=400, height=20)
    frame7.pack(side="bottom", pady=10)
    frame6.pack(side="bottom", pady=10)
    frame5.pack(side="bottom", pady=10)
    frame4.pack(side="bottom", pady=10)
    frame2.pack(side="right")
    frame1.pack(side="top")
    frame3.pack(side="top")


	# まずはスペクトログラムを描画
    fig = plt.figure()
    ax11 = fig.add_subplot(3, 1, 1)
    ax12 = fig.add_subplot(3, 1, 2)
    ax13 = fig.add_subplot(3, 1, 3)
    canvas = FigureCanvasTkAgg(fig, master=frame1)	# masterに対象とするframeを指定
    ax11.set_xlabel('sec')
    ax11.set_ylabel('frequency [Hz]')
    ax11.imshow(
		np.flipud(np.array(spectrogram4).T),
		extent=[0, duration, 0, 2000],
		aspect='auto',
		interpolation='nearest'
    )
    x_data = [xx * size_shift / SR for xx in range(len(spectrogram))] 
    ax12.set_xlabel('time[s]')
    ax12.set_ylabel('frequency [Hz]')
    ax12.set_ylim(0, 500)
    ax12.set_xlim(0, duration)
    ax12.plot(x_data, basics)
    ax13.set_xlabel('time[s]')					# x軸のラベルを設定
    ax13.set_ylabel('judge')		# y軸のラベルを設定
    ax13.set_ylim(-1, 5)
    ax13.set_xlim(0, duration)
    ax13.plot(x_data, sikibetu)
    canvas.get_tk_widget().pack(side="left")	# 最後にFrameに追加する処理

	# スライドバーの値が変更されたときに呼び出されるコールバック関数
	# ここで右側のグラフに
	# vはスライドバーの値
    def _draw_spectrum(v):
        SR = 16000
        # スライドバーの値からスペクトルのインデクスおよびそのスペクトルを取得
        index = int((len(spectrogram)-1) * (float(v) / duration))
        spectrum = spectrogram[index]

        # 直前のスペクトル描画を削除し，新たなスペクトルを描画
        ax2.clear()
        x_data = np.fft.rfftfreq(size_frame, d=1/SR)
        ax2.plot(x_data, spectrum)
        ax2.set_ylim(-10, 5)
        ax2.set_xlim(0, SR/2)
        ax2.set_ylabel('amblitude')
        ax2.set_xlabel('frequency [Hz]')
        canvas2.draw()
        numtext1.set("音量：%f [dB]"%(voll[index]))
        numtext2.set("基本周波数：%f [Hz]"%(basics[index]))
        mother = "無し"
        if sikibetu[index] == 0:
            mother = "あ"
        elif sikibetu[index] == 1:
            mother = "い"
        elif sikibetu[index] == 2:
            mother = "う"
        elif sikibetu[index] == 3:
            mother = "え"
        elif sikibetu[index] == 4:
            mother = "お"
        numtext3.set("母音推定：" + mother)

	# スペクトルを表示する領域を確保
	# ax2, canvs2 を使って上記のコールバック関数でグラフを描画する
    fig2, ax2 = plt.subplots()
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.get_tk_widget().pack(side="top")	# "top"は上部方向にウィジェットを積むことを意味する

	
	
	

	# スライドバーを作成
    scale = tkinter.Scale(
		command=_draw_spectrum,		# ここにコールバック関数を指定
		master=frame2,				# 表示するフレーム
		from_=0,					# 最小値
		to=duration,				# 最大値
		resolution=size_shift/SR,	# 刻み幅
		label=u'時間[sec]',
		orient=tkinter.HORIZONTAL,	# 横方向にスライド
		length=300,					# 横サイズ
		width=15,					# 縦サイズ
		font=("", 10)				# フォントサイズは20pxに設定
    )
    scale.pack(side="left")
    numtext1 = tkinter.StringVar(frame2)
    numtext1.set("音量：")
    label1 = tkinter.Label(master=frame2, textvariable=numtext1)
    label1.pack(side="top")
    numtext2 = tkinter.StringVar(frame2)
    numtext2.set("基本周波数：")
    label2 = tkinter.Label(master=frame2, textvariable=numtext2)
    label2.pack(side="top")
    numtext3 = tkinter.StringVar(frame2)
    numtext3.set("母音推定：")
    label3 = tkinter.Label(master=frame2, textvariable=numtext3)
    label3.pack(side="top")

    label11 = tkinter.Label(frame3, text=filename)
    label11.pack(side="top", pady=5)

    pygame.mixer.init(frequency = SR)
    global flag, s_flag
    flag = 0
    def play():
        global flag
        if flag == 1:
            pygame.mixer.music.unpause()
            flag = 2
        else:
            pygame.mixer.music.load(filename)     # 音楽ファイルの読み込み
            pygame.mixer.music.play()
            flag = 2

    def pause():
        global flag
        if flag == 2:
            pygame.mixer.music.pause()
            flag = 1

    def stop():
        global flag
        pygame.mixer.music.stop()
        flag = 0
        
    def chr():
        chroma_display(filename)

    def voice_c():
        pygame.mixer.music.stop()
        fre = scaleF.get()
        sin_wave = generate_sinusoid(SR, fre, duration)

        # 最大値を0.9にする
        sin_wave = sin_wave * 0.9

        # 元の音声と正弦波を重ね合わせる
        x_changed = x * sin_wave

        # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
        x_changed = (x_changed * 32768.0). astype('int16')

        # 音声ファイルとして出力する
        nowtime = time.time()
        fname = 'voice_controll/' + str(nowtime) + 'voicechange.wav'
        scipy.io.wavfile.write(fname , int(SR), x_changed)
        pygame.mixer.music.load(fname)     # 音楽ファイルの読み込み
        pygame.mixer.music.play()

    def tremolo():
        pygame.mixer.music.stop()
        Dval = scaleD.get()
        Rval = scaleR.get()
        sin_wave = 1 + Dval * generate_sinusoid(SR, Rval/SR, duration)

        # 最大値を0.9にする
        sin_wave = sin_wave * 0.9

        # 元の音声と正弦波を重ね合わせる
        x_changed = x * sin_wave

        # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
        x_changed = (x_changed * 32768.0). astype('int16')

        # 音声ファイルとして出力する
        nowtime = time.time()
        fname = 'voice_controll/' + str(nowtime) + 'tremolo.wav'
        scipy.io.wavfile.write(fname , int(SR), x_changed)
        pygame.mixer.music.load(fname)     # 音楽ファイルの読み込み
        pygame.mixer.music.play()

    def vibrato():
        pygame.mixer.music.stop()
        Dval = scaleD2.get()
        Rval = scaleR2.get()
        sin_wave = Dval * generate_sinusoid(SR, Rval/SR, duration)

        # 最大値を0.9にする
        sin_wave = sin_wave * 0.9

        # 元の音声と正弦波を重ね合わせる
        x_changed = [x[0] if i - int(sin_wave[i]) < 0 else (x[len(x)-1] if i - int(sin_wave[i]) > len(x)-1
            else x[i - int(sin_wave[i])]) for i in range(len(x))]

        # 値の範囲を[-1.0 ~ +1.0] から [-32768 ~ +32767] へ変換する
        x_changed = (np.array(x_changed) * 32768.0). astype('int16')

        # 音声ファイルとして出力する
        nowtime = time.time()
        fname = 'voice_controll/' + str(nowtime) + 'vibrato.wav'
        scipy.io.wavfile.write(fname , int(SR), x_changed)
        pygame.mixer.music.load(fname)     # 音楽ファイルの読み込み
        pygame.mixer.music.play()

    def echo():
        pygame.mixer.music.stop()
        number = scaleD3.get()
        times = scaleD4.get()
        drate = scaleR3.get()
        x_changed = x
        for ii in range(number):
            rolling = np.roll(x, int(SR*times)*(ii+1)) * (drate**(ii+1))
            x_changed = x_changed + [0 if iii < int(SR*times)*(ii+1) else rolling[iii] for iii in range(len(rolling))]
        x_changed = (np.array(x_changed) * 32768.0). astype('int16')
        nowtime = time.time()
        fname = 'voice_controll/' + str(nowtime) + 'echo.wav'
        scipy.io.wavfile.write(fname , int(SR), x_changed)
        pygame.mixer.music.load(fname)     # 音楽ファイルの読み込み
        pygame.mixer.music.play()


    button1 = tkinter.Button(frame3, text="再生", command=play)
    button1.pack(side="left", padx=5)

    button11 = tkinter.Button(frame3, text="一時停止", command=pause)
    button11.pack(side="left", padx=5)

    button111 = tkinter.Button(frame3, text="終了", command=stop)
    button111.pack(side="left", padx=5)




    button2 = tkinter.Button(frame3, text="クロマグラム表示", fg='#ffffff', bg='#ff0000', command=chr)
    button2.pack(side="left", padx=50)

    button31 = tkinter.Button(frame4, text="ボイスチェンジ", command=voice_c)
    button31.pack(side="right", padx=20)

    label31 = tkinter.Label(frame4, text="周波数:")
    label31.pack(side="left", pady=5)

    scaleF = tkinter.Scale(frame4, orient=tkinter.HORIZONTAL, length = 800, width = 5, from_ = 100, to = 1000)
    scaleF.pack(side="left")

    

    button32 = tkinter.Button(frame5, text="トレモロ", command=tremolo)
    button32.pack(side="right", padx=20)

    label32 = tkinter.Label(frame5, text="D:")
    label32.pack(side="left", pady=5)

    scaleD = tkinter.Scale(frame5, orient=tkinter.HORIZONTAL, length = 400, width = 5, from_ = 1, to = 50)
    scaleD.pack(side="left")

    label33 = tkinter.Label(frame5, text="R:")
    label33.pack(side="left", pady=5)

    scaleR = tkinter.Scale(frame5, orient=tkinter.HORIZONTAL, length = 400, width = 5, from_ = 8000, to = 320000, resolution=1000)
    scaleR.pack(side="left")

    button33 = tkinter.Button(frame6, text="ビブラート", command=vibrato)
    button33.pack(side="right", padx=20)

    label32 = tkinter.Label(frame6, text="D:")
    label32.pack(side="left", pady=5)

    scaleD2 = tkinter.Scale(frame6, orient=tkinter.HORIZONTAL, length = 400, width = 5, from_ = 1, to = 500)
    scaleD2.pack(side="left")

    label33 = tkinter.Label(frame6, text="R:")
    label33.pack(side="left", pady=5)

    scaleR2 = tkinter.Scale(frame6, orient=tkinter.HORIZONTAL, length = 400, width = 5, from_ = 8000, to = 320000, resolution=1000)
    scaleR2.pack(side="left")

    button34 = tkinter.Button(frame7, text="ディレイ・エコー", command=echo)
    button34.pack(side="right", padx=20)

    label34 = tkinter.Label(frame7, text="回数:")
    label34.pack(side="left", pady=5)

    scaleD3 = tkinter.Scale(frame7, orient=tkinter.HORIZONTAL, length = 240, width = 5, from_ = 1, to = 10)
    scaleD3.pack(side="left")

    label36 = tkinter.Label(frame7, text="間隔:")
    label36.pack(side="left", pady=5)

    scaleD4 = tkinter.Scale(frame7, orient=tkinter.HORIZONTAL, length = 240, width = 5, from_ =0.01, to = 1, resolution=0.01)
    scaleD4.pack(side="left")


    label35 = tkinter.Label(frame7, text="減衰率:")
    label35.pack(side="left", pady=5)

    scaleR3 = tkinter.Scale(frame7, orient=tkinter.HORIZONTAL, length = 240, width = 5, from_ = 0.01, to = 1, resolution=0.01)
    scaleR3.pack(side="left")

    # TkinterのGUI表示を開始

    root.mainloop()

# サンプリングレート
SR = 16000

# 音声ファイルの読み込み
x_a, _ = librosa.load('output_tan_a.wav', sr=SR)
x_i, _ = librosa.load('output_tan_i.wav', sr=SR)
x_u, _ = librosa.load('output_tan_u.wav', sr=SR)
x_e, _ = librosa.load('input_e.wav', sr=SR)
x_o, _ = librosa.load('input_o.wav', sr=SR)
xx = [x_a, x_i, x_u, x_e, x_o]
size_frame = 512
hamming_window = np.hamming(size_frame)
size_shift = 16000 / 100
myu = []
sigma = []
for j in range(0,5):
    ceps = []
    for i in np.arange(0, len(xx[j])-size_frame, size_shift):
        idx = int(i)
        x_frame = xx[j][idx : idx+size_frame]
        fft_spec = np.fft.rfft(x_frame * hamming_window)

        # 複素スペクトルを対数振幅スペクトルに
        fft_log_abs_spec = np.log(np.abs(fft_spec))

        cep = cepstrum(np.abs(fft_spec))

        ceps.append([np.real(cep[ii]) for ii in range(0,13)])

    myu_d = [sum(index) / len(ceps) for index in zip(*ceps)]
    myu.append(myu_d)
    sigma_d = [[(index[ii] - myu_d[ii])**2 for ii in range(0,13)] for index in ceps]
    sigma.append([sum(index) / len(sigma_d) for index in zip(*sigma_d)])

shutil.rmtree('voice_controll')
os.mkdir('voice_controll')

#ウインドウの作成
root0 = tkinter.Tk()
root0.title("Python GUI")
root0.geometry("360x100")

#入力欄の作成
input_box = tkinter.Entry(width=40)
input_box.place(x=20, y=50)

#ラベルの作成
input_label = tkinter.Label(text="音声ファイルを選択してください")
input_label.place(x=20, y=20)

#ボタンの作成
button = tkinter.Button(text="参照",command=file_select)
button.place(x=270, y=47)


root0.protocol("WM_DELETE_WINDOW", click_close)
root0.mainloop()
