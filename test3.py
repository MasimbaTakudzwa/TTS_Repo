from tts_pipeline_free import TTSPipeline

pipe = TTSPipeline()
pipe.speak_text("This is a test of the TTS pipeline.", output_path="test.wav")