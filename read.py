from tts_pipeline_free import TTSPipeline

pipe = TTSPipeline(reference_audio="my_voice.mp3")  # any format now works
pipe.speak_text("'Hell is full of good intentions or desires', is a historic aphorism also commonly cited these days as 'The road to hell is paved with good intentions'.The origin of this infamous proverb is attributed to Saint Bernard, a 12th-century French abbot.", output_path="cloned.wav")