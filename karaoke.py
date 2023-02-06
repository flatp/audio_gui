import pyaudio
import numpy as np
import threading
import time
import math

# matplotlib関連
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# GUI関連
import tkinter
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import (
	FigureCanvasTkAgg, NavigationToolbar2Tk)

# mp3ファイルを読み込んで再生
from pydub import AudioSegment
from pydub.utils import make_chunks

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

def hz2nn(frequency):
	try:
		out = int (round (12.0 * (math.log(frequency / 440.0) / math.log (2.0)))) + 69
	except:
		out = -1
	return out

def file_select():
	global audio_data, stream_play, p_play, p, format, channels, rate, output
	idir = 'C:\\python_test' #初期フォルダ
	filetype = [("音楽","*.wav")] #拡張子の選択
	filename = tkinter.filedialog.askopenfilename(filetypes = filetype, initialdir = idir)
	# pydubを使用して音楽ファイルを読み込む
	audio_data = AudioSegment.from_mp3(filename)
	stream_play = p_play.open(
		format = p.get_format_from_width(audio_data.sample_width),	# ストリームを読み書きするときのデータ型
		channels = audio_data.channels,								# チャネル数
		rate = audio_data.frame_rate,								# サンプリングレート
		output = True												# 出力モードに設定
	)

# サンプリングレート
SAMPLING_RATE = 16000

# フレームサイズ
FRAME_SIZE = 2048

# サイズシフト
SHIFT_SIZE = int(SAMPLING_RATE / 20)	# 今回は0.05秒
SPECTRUM_MIN = -5
SPECTRUM_MAX = 1

# 音量を表示する際の値の範囲
VOLUME_MIN = -120
VOLUME_MAX = -10

# log10を計算する際に，引数が0にならないようにするためにこの値を足す
EPSILON = 1e-10

# ハミング窓
hamming_window = np.hamming(FRAME_SIZE)

# グラフに表示する縦軸方向のデータ数
MAX_NUM_SPECTROGRAM = int(FRAME_SIZE / 2)

# グラフに表示する横軸方向のデータ数
NUM_DATA_SHOWN = 100

# GUIの開始フラグ（まだGUIを開始していないので、ここではFalseに）
is_gui_running = False

# matplotlib animation によって呼び出される関数
# ここでは最新のスペクトログラムと音量のデータを格納する
# 再描画はmatplotlib animationが行う
def animate(frame_index):

	ax1_sub.set_array(spectrogram_data)
	
	ax2_sub.set_data(time_x_data, volume_data)
	
	return ax1_sub, ax2_sub

def animate2(frame_index):

	ax3_sub.set_array(note_data)
	
	ax4_sub.set_data(time_x_data, basics)
	
	return ax3_sub, ax4_sub

def animate3(frame_index):

	ax5_sub.set_data(x_data, fft_log_abs_spec)
	
	return ax5_sub

def animate4(frame_index):

	ax6_sub.set_array(spectrogram_data_music)
	
	return ax6_sub

# GUIで表示するための処理（Tkinter）
root = tkinter.Tk()
root.wm_title("EXP4-AUDIO-SAMPLE")

frame1 = tkinter.Frame(root)
frame2 = tkinter.Frame(root)
frame3 = tkinter.Frame(root)
frame4 = tkinter.Frame(root)
frame3.pack(side="bottom")
frame1.pack(side="left")
frame2.pack(side="left")
frame4.pack(side="left")
frame5 = tkinter.Frame(frame3)
frame6 = tkinter.Frame(frame3)
frame7 = tkinter.Frame(frame3)
frame8 = tkinter.Frame(frame3)
frame9 = tkinter.Frame(frame3)
frame9.pack(side="left")
frame5.pack(side="top", pady=60)
frame7.pack(side="right", padx=200)
frame8.pack(side="top", padx=190, pady=20)
frame6.pack(side="top", padx=190, pady=20)


# スペクトログラムを描画
fig, ax1 = plt.subplots(1, 1)
canvas = FigureCanvasTkAgg(fig, master=frame1)

# 横軸の値のデータ
time_x_data = np.linspace(0, NUM_DATA_SHOWN * (SHIFT_SIZE/SAMPLING_RATE), NUM_DATA_SHOWN)
# 縦軸の値のデータ
freq_y_data = np.linspace(8000/MAX_NUM_SPECTROGRAM, 8000, MAX_NUM_SPECTROGRAM)

# とりあえず初期値（ゼロ）のスペクトログラムと音量のデータを作成
# この numpy array にデータが更新されていく
spectrogram_data = np.zeros((len(freq_y_data), len(time_x_data)))
volume_data = np.zeros(len(time_x_data))

# 楽曲のスペクトログラムを格納するデータ（このサンプルでは計算のみ）
spectrogram_data_music = np.zeros((len(freq_y_data), len(time_x_data)))

X = np.zeros(spectrogram_data.shape)
Y = np.zeros(spectrogram_data.shape)
for idx_f, f_v in enumerate(freq_y_data):
	for idx_t, t_v in enumerate(time_x_data):
		X[idx_f, idx_t] = t_v
		Y[idx_f, idx_t] = f_v

# pcolormeshを用いてスペクトログラムを描画
# 戻り値はデータの更新 & 再描画のために必要
ax1_sub = ax1.pcolormesh(
	X,
	Y,
	spectrogram_data,
	shading='nearest',	# 描画スタイル
	cmap='jet',			# カラーマップ
	norm=Normalize(SPECTRUM_MIN, SPECTRUM_MAX)	# 値の最小値と最大値を指定して，それに色を合わせる
)

# 音量を表示するために反転した軸を作成
ax2 = ax1.twinx()

# 音量をプロットする
# 戻り値はデータの更新 & 再描画のために必要
ax2_sub, = ax2.plot(time_x_data, volume_data)

# ラベルの設定
ax1.set_xlabel('sec')				# x軸のラベルを設定
ax1.set_ylabel('frequency [Hz]')	# y軸のラベルを設定
ax2.set_ylabel('volume [dB]')		# 反対側のy軸のラベルを設定

# 音量を表示する際の値の範囲を設定
ax2.set_ylim([VOLUME_MIN, VOLUME_MAX])

# maplotlib animationを設定
ani = animation.FuncAnimation(
	fig,
	animate,		# 再描画のために呼び出される関数
	interval=200,	# 100ミリ秒間隔で再描画を行う（PC環境によって処理が追いつかない場合はこの値を大きくするとよい）
	blit=True		# blitting処理を行うため描画処理が速くなる
)

fig2, ax3 = plt.subplots(1, 1)
canvas2 = FigureCanvasTkAgg(fig2, master=frame2)

# 縦軸の値のデータ
# freq_y_data2 = np.linspace(8000/MAX_NUM_SPECTROGRAM, 2000, MAX_NUM_SPECTROGRAM//4)
note_data = np.zeros((12, len(time_x_data)))
note_y_data = np.array([i for i in range(12)])
# spectrogram_data2 = np.zeros((len(freq_y_data2), len(time_x_data)))
# 楽曲のスペクトログラムを格納するデータ（このサンプルでは計算のみ）
basics = np.zeros(len(time_x_data))

X2 = np.zeros(note_data.shape)
Y2 = np.zeros(note_data.shape)
for idx_f, f_v in enumerate(note_y_data):
	for idx_t, t_v in enumerate(time_x_data):
		X2[idx_f, idx_t] = t_v
		Y2[idx_f, idx_t] = f_v

# pcolormeshを用いてスペクトログラムを描画
# 戻り値はデータの更新 & 再描画のために必要
ax3_sub = ax3.pcolormesh(
	X2,
	Y2,
	note_data,
	shading='nearest',	# 描画スタイル
	cmap='bwr',			# カラーマップ
	norm=Normalize(-1.5, 1.5)	# 値の最小値と最大値を指定して，それに色を合わせる
)

# 音量を表示するために反転した軸を作成
ax4 = ax3.twinx()

# 音量をプロットする
# 戻り値はデータの更新 & 再描画のために必要
ax4_sub, = ax4.plot(time_x_data, basics)

ax3.set_xlabel('sec')				# x軸のラベルを設定
ax3.set_ylabel('note_number')	# y軸のラベルを設定
ax4.set_ylabel('frequency [Hz]')	# y軸のラベルを設定
ax4.set_ylim([0, 300])
ani2 = animation.FuncAnimation(
	fig2,
	animate2,		# 再描画のために呼び出される関数
	interval=200,	# 100ミリ秒間隔で再描画を行う（PC環境によって処理が追いつかない場合はこの値を大きくするとよい）
	blit=True		# blitting処理を行うため描画処理が速くなる
)

fig3, ax5 = plt.subplots(1, 1)
canvas3 = FigureCanvasTkAgg(fig3, master=frame4)
fft_log_abs_spec = np.zeros(len(freq_y_data))
x_data = np.linspace((SAMPLING_RATE/2)/len(freq_y_data), SAMPLING_RATE/2, len(freq_y_data))
ax5_sub, = ax5.plot(x_data, fft_log_abs_spec, c='g')
ax5.set_xlabel('frequency [Hz]')
ax5.set_ylabel('amblitude')
ax5.set_xlim([0, 2000])
ax5.set_ylim([-4, 4])
ani3 = animation.FuncAnimation(
	fig3,
	animate3,		# 再描画のために呼び出される関数
	interval=500,	# 100ミリ秒間隔で再描画を行う（PC環境によって処理が追いつかない場合はこの値を大きくするとよい）
)

fig4, ax6 = plt.subplots(1, 1)
canvas4 = FigureCanvasTkAgg(fig4, master=frame9)
ax6_sub = ax6.pcolormesh(
	X,
	Y,
	spectrogram_data_music,
	shading='nearest',	# 描画スタイル
	cmap='jet',			# カラーマップ
	norm=Normalize(SPECTRUM_MIN, SPECTRUM_MAX)	# 値の最小値と最大値を指定して，それに色を合わせる
)
ax6.set_xlabel('sec')				# x軸のラベルを設定
ax6.set_ylabel('frequency [Hz]')	# y軸のラベルを設定
ani4 = animation.FuncAnimation(
	fig4,
	animate4,		# 再描画のために呼び出される関数
	interval=200,	# 100ミリ秒間隔で再描画を行う（PC環境によって処理が追いつかない場合はこの値を大きくするとよい）
	# blit=True		# blitting処理を行うため描画処理が速くなる
)

# matplotlib を GUI(Tkinter) に追加する
canvas.get_tk_widget().pack()
canvas2.get_tk_widget().pack()
canvas3.get_tk_widget().pack()
canvas4.get_tk_widget().pack()

# 再生位置をテキストで表示するためのラベルを作成
text = tkinter.StringVar()
text.set('0.0')
label = tkinter.Label(master=frame6, textvariable=text, font=("", 30))
label.pack(side="left", padx=30)
numtext1 = tkinter.StringVar(frame5)
numtext1.set("音量：")
label1 = tkinter.Label(master=frame5, textvariable=numtext1, font=("", 20))
label1.pack(side="top")
numtext2 = tkinter.StringVar(frame5)
numtext2.set("基本周波数：")
label2 = tkinter.Label(master=frame5, textvariable=numtext2, font=("", 20))
label2.pack(side="top")

# 終了ボタンが押されたときに呼び出される関数
# ここではGUIを終了する
def _quit():
	root.quit()
	root.destroy()

def _play():
	t_play_music = threading.Thread(target=play_music)
	t_play_music.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため
	t_play_music.start()


# 終了ボタンを作成
button1 = tkinter.Button(frame8, text="音声ファイル選択", command=file_select, font=("", 30))
button1.pack(side="left", padx=30)
button0 = tkinter.Button(master=frame6, text="▶", command=_play, font=("", 30))
button0.pack(side="left", padx=50)
button = tkinter.Button(master=frame7, text="終了", command=_quit, font=("", 30))
button.pack()


#
# (2) マイク入力のための処理
#

x_stacked_data = np.array([])

# フレーム毎に呼び出される関数
def input_callback(in_data, frame_count, time_info, status_flags):
	global x_stacked_data, spectrogram_data, volume_data, basics, fft_log_abs_spec, note_data, numtext1, numtext2

	# 現在のフレームの音声データをnumpy arrayに変換
	x_current_frame = np.frombuffer(in_data, dtype=np.float32)

	# 現在のフレームとこれまでに入力されたフレームを連結
	x_stacked_data = np.concatenate([x_stacked_data, x_current_frame])

	# フレームサイズ分のデータがあれば処理を行う
	if len(x_stacked_data) >= FRAME_SIZE:
		
		# フレームサイズからはみ出した過去のデータは捨てる
		x_stacked_data = x_stacked_data[len(x_stacked_data)-FRAME_SIZE:]

		# スペクトルを計算
		fft_spec = np.fft.rfft(x_stacked_data * hamming_window)
		fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]
		autocorr = np.correlate(x_stacked_data * hamming_window, x_stacked_data * hamming_window, 'full')
		autocorr = autocorr [len (autocorr ) // 2 : ]
		peakindices = [i for i in range (len (autocorr )) if is_peak (autocorr, i)]
		peakindices = [i for i in peakindices if i != 0]

		# ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
		# 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
		spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
		spectrogram_data[:, -1] = fft_log_abs_spec

		# 音量も同様の処理
		vol = 20 * np.log10(np.mean(x_current_frame ** 2) + EPSILON)
		volume_data = np.roll(volume_data, -1)
		volume_data[-1] = vol

		zero = zero_cross(x_stacked_data)
		basics = np.roll(basics, -1)
		try:
			max_peak_index = max (peakindices , key=lambda index: autocorr [index])
			if 16000 / max_peak_index < 300 and zero < 150 and vol > -80:
				freq_now = 16000 / max_peak_index
			else:
				freq_now = 0
		except:
			freq_now = 0

		basics[-1] = freq_now
		note = hz2nn(freq_now)
		if note == -1:
			notenum = np.zeros(12)
		else:
			note = note % 12
			notenum = np.array([1 if i == note else 0 for i in range(12)])

		note_data = np.roll(note_data, -1, axis=1)
		note_data[:, -1] = notenum

		numtext1.set("音量：%f [dB]"%(vol))
		numtext2.set("基本周波数：%f [Hz]"%(basics[-1]))
	
	# 戻り値は pyaudio の仕様に従うこと
	return None, pyaudio.paContinue

# マイクからの音声入力にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
p = pyaudio.PyAudio()
stream = p.open(
	format = pyaudio.paFloat32,
	channels = 1,
	rate = SAMPLING_RATE,
	input = True,						# ここをTrueにするとマイクからの入力になる 
	frames_per_buffer = SHIFT_SIZE,		# シフトサイズ
	stream_callback = input_callback	# ここでした関数がマイク入力の度に呼び出される（frame_per_bufferで指定した単位で）
)


#
# (3) mp3ファイル音楽を再生する処理
#

# mp3ファイル名
# ここは各自の音源ファイルに合わせて変更すること
filename = 'output_all.wav'

# pydubを使用して音楽ファイルを読み込む
audio_data = AudioSegment.from_mp3(filename)

# 音声ファイルの再生にはpyaudioを使用
# ここではpyaudioの再生ストリームを作成
p_play = pyaudio.PyAudio()
stream_play = p_play.open(
	format = p.get_format_from_width(audio_data.sample_width),	# ストリームを読み書きするときのデータ型
	channels = audio_data.channels,								# チャネル数
	rate = audio_data.frame_rate,								# サンプリングレート
	output = True												# 出力モードに設定
)

# 楽曲のデータを格納
x_stacked_data_music = np.array([])

# pydubで読み込んだ音楽ファイルを再生する部分のみ関数化する
def play_music():

	# この関数は別スレッドで実行するため
	# メインスレッドで定義した以下の２つの変数を利用できるように global 宣言する
	global is_gui_running, audio_data, now_playing_sec, x_stacked_data_music, spectrogram_data_music

	# pydubのmake_chunksを用いて音楽ファイルのデータを切り出しながら読み込む
	# 第二引数には何ミリ秒毎に読み込むかを指定
	# ここでは10ミリ秒ごとに読み込む
	

	size_frame_music = 10	# 10ミリ秒毎に読み込む

	idx = 0

	for chunk in make_chunks(audio_data, size_frame_music):
		
		# GUIが終了してれば，この関数の処理も終了する
		if is_gui_running == False:
			break

		# pyaudioの再生ストリームに切り出した音楽データを流し込む
		# 再生が完了するまで処理はここでブロックされる
		stream_play.write(chunk._data)
		
		# 現在の再生位置を計算（単位は秒）
		now_playing_sec = (idx * size_frame_music) / 1000.
		
		idx += 1 
		
		# データの取得
		data_music = np.array(chunk.get_array_of_samples())
		
		# 正規化
		data_music = data_music / np.iinfo(np.int32).max	

		#
		# 以下はマイク入力のときと同様
		#

		# 現在のフレームとこれまでに入力されたフレームを連結
		x_stacked_data_music = np.concatenate([x_stacked_data_music, data_music])

		# フレームサイズ分のデータがあれば処理を行う
		if len(x_stacked_data_music) >= FRAME_SIZE:
			
			# フレームサイズからはみ出した過去のデータは捨てる
			x_stacked_data_music = x_stacked_data_music[len(x_stacked_data_music)-FRAME_SIZE:]

			# スペクトルを計算
			fft_spec = np.fft.rfft(x_stacked_data_music * hamming_window)
			fft_log_abs_spec = np.log10(np.abs(fft_spec) + EPSILON)[:-1]

			# ２次元配列上で列方向（時間軸方向）に１つずらし（戻し）
			# 最後の列（＝最後の時刻のスペクトルがあった位置）に最新のスペクトルデータを挿入
			spectrogram_data_music = np.roll(spectrogram_data_music, -1, axis=1)
			spectrogram_data_music[:, -1] = fft_log_abs_spec

# 再生時間の表示を随時更新する関数
def update_gui_text():

	global is_gui_running, now_playing_sec, text

	while True:
		
		# GUIが表示されていれば再生位置（秒）をテキストとしてGUI上に表示
		if is_gui_running:
			text.set('%.3f' % now_playing_sec)
		
		# 0.01秒ごとに更新
		time.sleep(0.01)

# 再生時間を表す
now_playing_sec = 0.0

# 再生時間の表示を随時更新する関数を別スレッドで開始
t_update_gui = threading.Thread(target=update_gui_text)
t_update_gui.setDaemon(True)	# GUIが消されたときにこの別スレッドの処理も終了されるようにするため

#
# (4) 全体の処理を実行
#

# GUIの開始フラグをTrueに
is_gui_running = True

# 上記で設定したスレッドを開始（直前のフラグを立ててから）
# t_play_music.start()
t_update_gui.start()

# GUIを開始，GUIが表示されている間は処理はここでストップ
tkinter.mainloop()

# GUIの開始フラグをFalseに = 音楽再生スレッドのループを終了
is_gui_running = False

# 終了処理
stream_play.stop_stream()
stream_play.close()
p_play.terminate()
