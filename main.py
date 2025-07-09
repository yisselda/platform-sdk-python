"""
Creole Platform Python SDK

A comprehensive Python SDK for integrating with the Creole Translation Platform services.
Provides translation, speech-to-text, and text-to-speech capabilities.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Union, BinaryIO, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import aiohttp
import websockets
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CreoleSDKConfig:
    """Configuration for the Creole Platform SDK"""
    translation_url: str = "http://localhost:8001"
    stt_url: str = "http://localhost:8002"
    tts_url: str = "http://localhost:8003"
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class Language:
    """Represents a supported language"""
    code: str
    name: str
    native_name: str


@dataclass
class Voice:
    """Represents a text-to-speech voice"""
    id: str
    name: str
    language: str
    gender: str
    age: str
    description: str


@dataclass
class TranslationResult:
    """Result of a translation operation"""
    translated_text: str
    source_language: str
    target_language: str
    confidence: float


@dataclass
class TranscriptionResult:
    """Result of a speech-to-text operation"""
    text: str
    language: str
    confidence: float
    duration: float


@dataclass
class TranslationOptions:
    """Options for translation"""
    text: str
    source_language: str
    target_language: str


@dataclass
class BatchTranslationOptions:
    """Options for batch translation"""
    text: str
    source_language: str
    target_languages: List[str]


@dataclass
class TranscriptionOptions:
    """Options for transcription"""
    language: Optional[str] = None
    model: Optional[str] = None


@dataclass
class SynthesisOptions:
    """Options for speech synthesis"""
    language: str = "ht"
    voice: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0


@dataclass
class StreamingTranscriptionCallbacks:
    """Callbacks for streaming transcription"""
    on_partial_result: Optional[Callable[[str, float], None]] = None
    on_final_result: Optional[Callable[[TranscriptionResult], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None


class CreolePlatformSDK:
    """Main SDK class for interacting with the Creole Platform services"""
    
    def __init__(self, config: Optional[CreoleSDKConfig] = None):
        self.config = config or CreoleSDKConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._languages: Optional[List[Language]] = None
        self._voices: Optional[List[Voice]] = None
    
    async def __aenter__(self):
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _api_call(
        self, 
        method: str, 
        url: str, 
        data: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        retries: Optional[int] = None
    ) -> Dict:
        """Make an API call with retry logic"""
        await self._ensure_session()
        
        if retries is None:
            retries = self.config.retry_attempts
        
        for attempt in range(retries + 1):
            try:
                kwargs = {}
                if json_data:
                    kwargs['json'] = json_data
                if data:
                    kwargs['data'] = data
                if files:
                    kwargs['data'] = files
                
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        try:
                            error_data = json.loads(error_text)
                            error_msg = error_data.get('error', f'HTTP {response.status}')
                        except json.JSONDecodeError:
                            error_msg = f'HTTP {response.status}: {error_text}'
                        raise Exception(error_msg)
                        
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise e
    
    # Translation methods
    async def translate(self, options: TranslationOptions) -> TranslationResult:
        """Translate text from source to target language"""
        url = urljoin(self.config.translation_url, "/api/v1/translate")
        
        data = {
            "text": options.text,
            "source_language": options.source_language,
            "target_language": options.target_language
        }
        
        result = await self._api_call("POST", url, json_data=data)
        
        return TranslationResult(
            translated_text=result["translated_text"],
            source_language=result["source_language"],
            target_language=result["target_language"],
            confidence=result["confidence"]
        )
    
    async def translate_batch(self, options: BatchTranslationOptions) -> Dict[str, TranslationResult]:
        """Translate text to multiple target languages"""
        url = urljoin(self.config.translation_url, "/api/v1/translate/batch")
        
        data = {
            "text": options.text,
            "source_language": options.source_language,
            "target_languages": options.target_languages
        }
        
        result = await self._api_call("POST", url, json_data=data)
        
        translations = {}
        for lang, trans_data in result["translations"].items():
            translations[lang] = TranslationResult(
                translated_text=trans_data["translated_text"],
                source_language=trans_data["source_language"],
                target_language=trans_data["target_language"],
                confidence=trans_data["confidence"]
            )
        
        return translations
    
    async def get_supported_languages(self) -> List[Language]:
        """Get list of supported languages"""
        if self._languages is None:
            url = urljoin(self.config.translation_url, "/api/v1/languages")
            result = await self._api_call("GET", url)
            
            self._languages = [
                Language(
                    code=lang["code"],
                    name=lang["name"],
                    native_name=lang["native_name"]
                )
                for lang in result["supported_languages"]
            ]
        
        return self._languages
    
    # Speech-to-Text methods
    async def transcribe_audio(
        self, 
        audio_file: Union[str, Path, BinaryIO], 
        options: Optional[TranscriptionOptions] = None
    ) -> TranscriptionResult:
        """Transcribe audio file to text"""
        url = urljoin(self.config.stt_url, "/api/v1/transcribe")
        options = options or TranscriptionOptions()
        
        # Prepare file data
        if isinstance(audio_file, (str, Path)):
            with open(audio_file, 'rb') as f:
                file_data = f.read()
            filename = Path(audio_file).name
        else:
            file_data = audio_file.read()
            filename = "audio.wav"
        
        # Prepare form data
        files = {
            'file': (filename, file_data, 'audio/wav')
        }
        
        if options.language:
            files['language'] = (None, options.language)
        if options.model:
            files['model'] = (None, options.model)
        
        result = await self._api_call("POST", url, files=files)
        
        return TranscriptionResult(
            text=result["text"],
            language=result["language"],
            confidence=result["confidence"],
            duration=result["duration"]
        )
    
    async def detect_language(self, audio_file: Union[str, Path, BinaryIO]) -> Dict[str, Union[str, float]]:
        """Detect language of audio file"""
        url = urljoin(self.config.stt_url, "/api/v1/detect-language")
        
        # Prepare file data
        if isinstance(audio_file, (str, Path)):
            with open(audio_file, 'rb') as f:
                file_data = f.read()
            filename = Path(audio_file).name
        else:
            file_data = audio_file.read()
            filename = "audio.wav"
        
        files = {
            'file': (filename, file_data, 'audio/wav')
        }
        
        result = await self._api_call("POST", url, files=files)
        
        return {
            "detected_language": result["detected_language"],
            "confidence": result["confidence"]
        }
    
    async def start_streaming_transcription(
        self,
        callbacks: StreamingTranscriptionCallbacks,
        options: Optional[TranscriptionOptions] = None
    ) -> 'StreamingTranscription':
        """Start streaming transcription session"""
        return StreamingTranscription(self, callbacks, options)
    
    # Text-to-Speech methods
    async def synthesize_text(
        self, 
        text: str, 
        options: Optional[SynthesisOptions] = None
    ) -> bytes:
        """Synthesize text to speech"""
        url = urljoin(self.config.tts_url, "/api/v1/synthesize")
        options = options or SynthesisOptions()
        
        data = {
            "text": text,
            "language": options.language,
            "voice": options.voice,
            "speed": options.speed,
            "pitch": options.pitch,
            "volume": options.volume
        }
        
        await self._ensure_session()
        
        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                return await response.read()
            else:
                error_text = await response.text()
                try:
                    error_data = json.loads(error_text)
                    error_msg = error_data.get('error', f'HTTP {response.status}')
                except json.JSONDecodeError:
                    error_msg = f'HTTP {response.status}: {error_text}'
                raise Exception(error_msg)
    
    async def get_available_voices(self, language: Optional[str] = None) -> List[Voice]:
        """Get available voices for text-to-speech"""
        if self._voices is None:
            url = urljoin(self.config.tts_url, "/api/v1/voices")
            result = await self._api_call("GET", url)
            
            self._voices = [
                Voice(
                    id=voice["id"],
                    name=voice["name"],
                    language=voice["language"],
                    gender=voice["gender"],
                    age=voice["age"],
                    description=voice["description"]
                )
                for voice in result["voices"]
            ]
        
        if language:
            return [voice for voice in self._voices if voice.language == language]
        
        return self._voices
    
    async def preview_voice(
        self, 
        voice_id: str, 
        language: str = "ht", 
        sample_text: str = "Bonjou, koman ou ye?"
    ) -> bytes:
        """Preview a voice with sample text"""
        url = urljoin(self.config.tts_url, "/api/v1/preview")
        
        data = {
            "voice_id": voice_id,
            "language": language,
            "text": sample_text
        }
        
        await self._ensure_session()
        
        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                return await response.read()
            else:
                error_text = await response.text()
                raise Exception(f"Voice preview failed: {response.status} - {error_text}")
    
    # Health check methods
    async def check_health(self) -> Dict[str, bool]:
        """Check health of all services"""
        async def check_service(url: str) -> bool:
            try:
                health_url = urljoin(url, "/health")
                await self._api_call("GET", health_url)
                return True
            except:
                return False
        
        results = await asyncio.gather(
            check_service(self.config.translation_url),
            check_service(self.config.stt_url),
            check_service(self.config.tts_url),
            return_exceptions=True
        )
        
        return {
            "translation": results[0] if not isinstance(results[0], Exception) else False,
            "stt": results[1] if not isinstance(results[1], Exception) else False,
            "tts": results[2] if not isinstance(results[2], Exception) else False
        }


class StreamingTranscription:
    """Handles streaming transcription via WebSocket"""
    
    def __init__(
        self, 
        sdk: CreolePlatformSDK, 
        callbacks: StreamingTranscriptionCallbacks,
        options: Optional[TranscriptionOptions] = None
    ):
        self.sdk = sdk
        self.callbacks = callbacks
        self.options = options or TranscriptionOptions()
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
    
    async def connect(self):
        """Connect to the streaming transcription service"""
        ws_url = self.sdk.config.stt_url.replace("http", "ws") + "/api/v1/stream"
        
        try:
            self.websocket = await websockets.connect(ws_url)
            self.is_connected = True
            
            # Send initial configuration
            if self.options.language or self.options.model:
                config = {
                    "language": self.options.language or "auto",
                    "model": self.options.model or "whisper-base"
                }
                await self.send_config(config)
            
            # Start listening for messages
            asyncio.create_task(self._listen_for_messages())
            
        except Exception as e:
            self.is_connected = False
            if self.callbacks.on_error:
                self.callbacks.on_error(e)
    
    async def _listen_for_messages(self):
        """Listen for incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if data["type"] == "partial_transcript" and self.callbacks.on_partial_result:
                    self.callbacks.on_partial_result(
                        data["data"]["text"], 
                        data["data"]["confidence"]
                    )
                elif data["type"] == "final_transcript" and self.callbacks.on_final_result:
                    result = TranscriptionResult(
                        text=data["data"]["text"],
                        language=data["data"]["language"],
                        confidence=data["data"]["confidence"],
                        duration=0
                    )
                    self.callbacks.on_final_result(result)
                elif data["type"] == "error" and self.callbacks.on_error:
                    self.callbacks.on_error(Exception(data["data"]["message"]))
                    
        except Exception as e:
            self.is_connected = False
            if self.callbacks.on_error:
                self.callbacks.on_error(e)
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk for transcription"""
        if self.websocket and self.is_connected:
            import base64
            
            message = {
                "type": "audio_chunk",
                "data": base64.b64encode(audio_data).decode('utf-8')
            }
            
            await self.websocket.send(json.dumps(message))
    
    async def send_config(self, config: Dict):
        """Send configuration to the streaming service"""
        if self.websocket and self.is_connected:
            message = {
                "type": "config",
                "data": json.dumps(config)
            }
            
            await self.websocket.send(json.dumps(message))
    
    async def stop(self):
        """Stop the streaming transcription"""
        if self.websocket and self.is_connected:
            message = {"type": "stop"}
            await self.websocket.send(json.dumps(message))
            await self.websocket.close()
            self.is_connected = False


# Convenience functions
async def create_sdk(config: Optional[CreoleSDKConfig] = None) -> CreolePlatformSDK:
    """Create and initialize a Creole Platform SDK instance"""
    sdk = CreolePlatformSDK(config)
    await sdk._ensure_session()
    return sdk


# Example usage
async def main():
    """Example usage of the Creole Platform SDK"""
    async with CreolePlatformSDK() as sdk:
        # Check health
        health = await sdk.check_health()
        print("Service health:", health)
        
        # Get supported languages
        languages = await sdk.get_supported_languages()
        print("Supported languages:", [lang.name for lang in languages])
        
        # Translate text
        translation_result = await sdk.translate(TranslationOptions(
            text="Hello, how are you?",
            source_language="en",
            target_language="ht"
        ))
        print("Translation:", translation_result.translated_text)
        
        # Get available voices
        voices = await sdk.get_available_voices("ht")
        print("Available voices:", [voice.name for voice in voices])
        
        # Synthesize speech
        audio_data = await sdk.synthesize_text(
            translation_result.translated_text,
            SynthesisOptions(language="ht", voice="default")
        )
        print(f"Generated audio: {len(audio_data)} bytes")


if __name__ == "__main__":
    asyncio.run(main())