import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
import librosa
from openai import OpenAI
from difflib import SequenceMatcher
import tempfile
import time

class EnglishPronunciationApp:
    def __init__(self):
        self.client = None
        self.recording = None
        self.recognized_text = ""
        self.original_text = ""
        self.original_audio = None
        self.original_sr = None

    def set_api_key(self, api_key):
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                self.client.models.list()
                return True
            except Exception as e:
                st.error(f"API 키 설정 실패: {str(e)}")
                return False
        return False

    def generate_speech(self, text):
        if text:
            self.original_text = text
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                tts.save(temp_audio_file.name)
                self.original_audio, self.original_sr = librosa.load(temp_audio_file.name)
                return temp_audio_file.name
        return None

    def record_audio(self):
        duration = 5
        fs = 44100
        st.write("5초 동안 녹음합니다.")
        for i in range(5, 0, -1):
            st.write(f"{i}초 남았습니다...")
            time.sleep(1)
        st.write("녹음 중...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        return recording, fs

    def transcribe_audio(self, audio_file):
        if not self.client:
            return None
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        return transcript.text

    def calculate_similarity(self, text1, text2):
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() * 100

    def analyze_pronunciation(self, original_text, recognized_text, similarity):
        if not self.client:
            return None
        prompt = f"""
        원본 텍스트: {original_text}
        인식된 텍스트: {recognized_text}
        텍스트 유사도: {similarity:.2f}%

        위의 두 텍스트를 비교하여 발음의 정확성을 분석해주세요. 다음 사항들을 고려해주세요:
        1. 단어의 누락 또는 추가
        2. 발음의 차이
        3. 강세와 억양의 문제
        4. 전반적인 유창성

        분석 결과를 한국어로 작성해주시고, 개선을 위한 구체적인 조언도 함께 제공해주세요.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes English pronunciation."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

def plot_waveform(audio_data, sr, title, figsize=(2.5, 1)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(len(audio_data)) / sr, audio_data)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="영어 발음 분석기", layout="wide")
    st.title("영어 발음 분석기")

    if 'app' not in st.session_state:
        st.session_state.app = EnglishPronunciationApp()

    col1, col2 = st.columns([3, 1])

    with col1:
        if 'api_key_set' not in st.session_state:
            st.session_state.api_key_set = False

        if not st.session_state.api_key_set:
            api_key = st.text_input("OpenAI API Key:", type="password")
            if st.button("API 키 설정"):
                if st.session_state.app.set_api_key(api_key):
                    st.session_state.api_key_set = True
                    st.success("API 키가 성공적으로 설정되었습니다.")
                else:
                    st.error("올바른 API 키를 입력해주세요.")
        else:
            st.success("API 키가 설정되었습니다.")

        text_input = st.text_input("영어 문장 입력:")
        if st.button("음성 생성"):
            audio_file = st.session_state.app.generate_speech(text_input)
            if audio_file:
                st.audio(audio_file, format="audio/mp3")
                st.session_state.original_waveform = plot_waveform(st.session_state.app.original_audio, st.session_state.app.original_sr, "원본 음성")
                st.pyplot(st.session_state.original_waveform)

        if st.button("녹음 시작"):
            with st.spinner("녹음 준비 중..."):
                recording, fs = st.session_state.app.record_audio()
            st.success("녹음 완료!")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                sf.write(temp_audio_file.name, recording, fs)
                st.audio(temp_audio_file.name, format="audio/wav")
                st.pyplot(plot_waveform(recording.flatten(), fs, "녹음된 음성"))

                with open(temp_audio_file.name, "rb") as audio_file:
                    if st.session_state.api_key_set:
                        try:
                            recognized_text = st.session_state.app.transcribe_audio(audio_file)
                            if recognized_text:
                                st.write(f"인식된 텍스트: {recognized_text}")

                                if st.session_state.app.original_text:
                                    similarity = st.session_state.app.calculate_similarity(st.session_state.app.original_text, recognized_text)
                                    analysis = st.session_state.app.analyze_pronunciation(st.session_state.app.original_text, recognized_text, similarity)
                                    
                                    st.session_state.analysis_result = {
                                        "similarity": similarity,
                                        "analysis": analysis
                                    }
                            else:
                                st.warning("음성 인식에 실패했습니다. 다시 시도해주세요.")
                        except Exception as e:
                            st.error(f"오류 발생: {str(e)}")
                    else:
                        st.warning("음성 인식 및 발음 분석을 위해 유효한 API 키를 설정해주세요.")

        if 'original_waveform' in st.session_state:
            st.pyplot(st.session_state.original_waveform)

    with col2:
        st.subheader("분석 결과")
        if 'analysis_result' in st.session_state:
            st.markdown(f"**텍스트 유사도**: {st.session_state.analysis_result['similarity']:.2f}%")
            st.markdown("**분석:**")
            st.markdown(st.session_state.analysis_result['analysis'])

if __name__ == "__main__":
    main()