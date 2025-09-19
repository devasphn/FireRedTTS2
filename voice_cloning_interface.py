#!/usr/bin/env python3
"""
Voice Cloning and Multi-Speaker Interface
Handles voice profile management, cloning, and multi-speaker dialogue generation
"""

import os
import uuid
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import gradio as gr

from fireredtts2.fireredtts2 import FireRedTTS2

logger = logging.getLogger(__name__)

@dataclass
class VoiceProfile:
    """Represents a voice profile for cloning"""
    profile_id: str
    name: str
    description: str
    reference_audio_path: str
    reference_text: str
    language: str
    gender: str
    age_range: str
    voice_characteristics: Dict[str, Any]
    created_at: datetime
    last_used: datetime
    usage_count: int
    quality_score: float
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "reference_audio_path": self.reference_audio_path,
            "reference_text": self.reference_text,
            "language": self.language,
            "gender": self.gender,
            "age_range": self.age_range,
            "voice_characteristics": self.voice_characteristics,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "usage_count": self.usage_count,
            "quality_score": self.quality_score,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        """Create from dictionary"""
        return cls(
            profile_id=data["profile_id"],
            name=data["name"],
            description=data["description"],
            reference_audio_path=data["reference_audio_path"],
            reference_text=data["reference_text"],
            language=data["language"],
            gender=data["gender"],
            age_range=data["age_range"],
            voice_characteristics=data["voice_characteristics"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]),
            usage_count=data["usage_count"],
            quality_score=data["quality_score"],
            is_active=data.get("is_active", True)
        )

@dataclass
class MultiSpeakerConfig:
    """Configuration for multi-speaker dialogue"""
    config_id: str
    name: str
    description: str
    speakers: List[Dict[str, Any]]  # List of speaker configurations
    dialogue_style: str
    turn_management: str
    voice_consistency: bool
    created_at: datetime
    is_active: bool = True

class VoiceQualityAnalyzer:
    """Analyzes voice quality for cloning suitability"""
    
    def __init__(self):
        self.min_duration = 3.0  # Minimum 3 seconds
        self.max_duration = 30.0  # Maximum 30 seconds
        self.target_sample_rate = 16000
    
    def analyze_reference_audio(self, audio_data: Tuple[int, np.ndarray], 
                              reference_text: str = "") -> Dict[str, Any]:
        """
        Analyze reference audio for voice cloning quality
        
        Returns:
            Dict with quality metrics and recommendations
        """
        sample_rate, audio_array = audio_data
        
        # Ensure mono audio
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Convert to float32
        if audio_array.dtype != np.float32:
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            else:
                audio_array = audio_array.astype(np.float32)
        
        duration = len(audio_array) / sample_rate
        
        # Basic quality metrics
        rms_energy = np.sqrt(np.mean(audio_array ** 2))
        peak_amplitude = np.max(np.abs(audio_array))
        dynamic_range = np.max(audio_array) - np.min(audio_array)
        
        # Signal-to-noise ratio estimation
        # Use the quietest 10% as noise floor
        sorted_abs = np.sort(np.abs(audio_array))
        noise_floor = np.mean(sorted_abs[:int(len(sorted_abs) * 0.1)])
        signal_level = rms_energy
        snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))
        
        # Clipping detection
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio_array) > clipping_threshold)
        clipping_percent = (clipped_samples / len(audio_array)) * 100
        
        # Silence detection
        silence_threshold = 0.01
        silent_samples = np.sum(np.abs(audio_array) < silence_threshold)
        silence_percent = (silent_samples / len(audio_array)) * 100
        
        # Frequency analysis
        fft = np.fft.fft(audio_array)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_array), 1/sample_rate)[:len(fft)//2]
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(magnitude)
        dominant_freq = freqs[dominant_freq_idx]
        
        # Spectral centroid (brightness measure)
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        # Quality scoring
        quality_factors = {
            "duration": self._score_duration(duration),
            "snr": self._score_snr(snr_db),
            "clipping": self._score_clipping(clipping_percent),
            "silence": self._score_silence(silence_percent),
            "dynamic_range": self._score_dynamic_range(dynamic_range),
            "energy": self._score_energy(rms_energy)
        }
        
        overall_quality = np.mean(list(quality_factors.values()))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            duration, snr_db, clipping_percent, silence_percent, 
            dynamic_range, rms_energy
        )
        
        return {
            "overall_quality": float(overall_quality),
            "duration_seconds": float(duration),
            "sample_rate": int(sample_rate),
            "rms_energy": float(rms_energy),
            "peak_amplitude": float(peak_amplitude),
            "snr_db": float(snr_db),
            "clipping_percent": float(clipping_percent),
            "silence_percent": float(silence_percent),
            "dynamic_range": float(dynamic_range),
            "dominant_frequency_hz": float(dominant_freq),
            "spectral_centroid_hz": float(spectral_centroid),
            "quality_factors": quality_factors,
            "recommendations": recommendations,
            "suitable_for_cloning": overall_quality >= 0.7
        }
    
    def _score_duration(self, duration: float) -> float:
        """Score audio duration (0-1, higher is better)"""
        if duration < self.min_duration:
            return duration / self.min_duration * 0.5
        elif duration > self.max_duration:
            return max(0.5, 1.0 - (duration - self.max_duration) / self.max_duration)
        else:
            return 1.0
    
    def _score_snr(self, snr_db: float) -> float:
        """Score signal-to-noise ratio"""
        if snr_db < 10:
            return 0.0
        elif snr_db > 40:
            return 1.0
        else:
            return (snr_db - 10) / 30
    
    def _score_clipping(self, clipping_percent: float) -> float:
        """Score clipping (lower is better)"""
        if clipping_percent > 5:
            return 0.0
        elif clipping_percent > 1:
            return 1.0 - (clipping_percent - 1) / 4
        else:
            return 1.0
    
    def _score_silence(self, silence_percent: float) -> float:
        """Score silence percentage"""
        if silence_percent > 50:
            return 0.0
        elif silence_percent > 20:
            return 1.0 - (silence_percent - 20) / 30
        else:
            return 1.0
    
    def _score_dynamic_range(self, dynamic_range: float) -> float:
        """Score dynamic range"""
        if dynamic_range < 0.1:
            return 0.0
        elif dynamic_range > 1.5:
            return 1.0
        else:
            return dynamic_range / 1.5
    
    def _score_energy(self, rms_energy: float) -> float:
        """Score RMS energy level"""
        if rms_energy < 0.01:
            return 0.0
        elif rms_energy > 0.5:
            return 0.5
        else:
            return rms_energy / 0.3
    
    def _generate_recommendations(self, duration: float, snr_db: float, 
                                clipping_percent: float, silence_percent: float,
                                dynamic_range: float, rms_energy: float) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if duration < self.min_duration:
            recommendations.append(f"Audio too short ({duration:.1f}s). Minimum {self.min_duration}s recommended.")
        elif duration > self.max_duration:
            recommendations.append(f"Audio too long ({duration:.1f}s). Maximum {self.max_duration}s recommended.")
        
        if snr_db < 20:
            recommendations.append(f"Low signal-to-noise ratio ({snr_db:.1f}dB). Record in quieter environment.")
        
        if clipping_percent > 1:
            recommendations.append(f"Audio clipping detected ({clipping_percent:.1f}%). Reduce recording volume.")
        
        if silence_percent > 30:
            recommendations.append(f"Too much silence ({silence_percent:.1f}%). Trim silent portions.")
        
        if rms_energy < 0.02:
            recommendations.append("Audio level too low. Increase recording volume or speak closer to microphone.")
        
        if dynamic_range < 0.2:
            recommendations.append("Low dynamic range. Ensure natural speech variation.")
        
        if not recommendations:
            recommendations.append("Audio quality is good for voice cloning!")
        
        return recommendations

class VoiceProfileManager:
    """Manages voice profiles for cloning"""
    
    def __init__(self, storage_dir: str = "/workspace/voice_profiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.profiles_dir = self.storage_dir / "profiles"
        self.audio_dir = self.storage_dir / "audio"
        self.profiles_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        self.quality_analyzer = VoiceQualityAnalyzer()
        
        # Load existing profiles
        self._load_profiles()
    
    def create_voice_profile(self, 
                           name: str,
                           description: str,
                           audio_data: Tuple[int, np.ndarray],
                           reference_text: str,
                           language: str = "English",
                           gender: str = "Unknown",
                           age_range: str = "Unknown") -> Tuple[bool, str, Optional[VoiceProfile]]:
        """
        Create a new voice profile from reference audio
        
        Returns:
            (success, message, profile)
        """
        
        # Analyze audio quality
        quality_analysis = self.quality_analyzer.analyze_reference_audio(
            audio_data, reference_text
        )
        
        if not quality_analysis["suitable_for_cloning"]:
            return (
                False, 
                f"Audio quality insufficient for cloning (score: {quality_analysis['overall_quality']:.2f}). "
                f"Recommendations: {'; '.join(quality_analysis['recommendations'])}",
                None
            )
        
        # Generate unique profile ID
        profile_id = str(uuid.uuid4())
        
        # Save reference audio
        audio_filename = f"{profile_id}_reference.wav"
        audio_path = self.audio_dir / audio_filename
        
        try:
            sample_rate, audio_array = audio_data
            
            # Ensure mono and proper format
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Convert to tensor and save
            audio_tensor = torch.from_numpy(audio_array).float()
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            torchaudio.save(str(audio_path), audio_tensor, sample_rate)
            
        except Exception as e:
            logger.error(f"Failed to save reference audio: {e}")
            return False, f"Failed to save reference audio: {e}", None
        
        # Extract voice characteristics from analysis
        voice_characteristics = {
            "dominant_frequency_hz": quality_analysis["dominant_frequency_hz"],
            "spectral_centroid_hz": quality_analysis["spectral_centroid_hz"],
            "dynamic_range": quality_analysis["dynamic_range"],
            "rms_energy": quality_analysis["rms_energy"],
            "sample_rate": quality_analysis["sample_rate"]
        }
        
        # Create voice profile
        current_time = datetime.now()
        profile = VoiceProfile(
            profile_id=profile_id,
            name=name,
            description=description,
            reference_audio_path=str(audio_path),
            reference_text=reference_text,
            language=language,
            gender=gender,
            age_range=age_range,
            voice_characteristics=voice_characteristics,
            created_at=current_time,
            last_used=current_time,
            usage_count=0,
            quality_score=quality_analysis["overall_quality"]
        )
        
        # Save profile
        self.voice_profiles[profile_id] = profile
        self._save_profile(profile)
        
        logger.info(f"Created voice profile: {name} ({profile_id})")
        return True, f"Voice profile '{name}' created successfully!", profile
    
    def get_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get a voice profile by ID"""
        return self.voice_profiles.get(profile_id)
    
    def list_profiles(self, active_only: bool = True) -> List[VoiceProfile]:
        """List all voice profiles"""
        profiles = list(self.voice_profiles.values())
        if active_only:
            profiles = [p for p in profiles if p.is_active]
        return sorted(profiles, key=lambda p: p.last_used, reverse=True)
    
    def update_profile_usage(self, profile_id: str):
        """Update profile usage statistics"""
        if profile_id in self.voice_profiles:
            profile = self.voice_profiles[profile_id]
            profile.usage_count += 1
            profile.last_used = datetime.now()
            self._save_profile(profile)
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete a voice profile"""
        if profile_id not in self.voice_profiles:
            return False
        
        profile = self.voice_profiles[profile_id]
        
        try:
            # Delete audio file
            if os.path.exists(profile.reference_audio_path):
                os.remove(profile.reference_audio_path)
            
            # Delete profile file
            profile_file = self.profiles_dir / f"{profile_id}.json"
            if profile_file.exists():
                profile_file.unlink()
            
            # Remove from memory
            del self.voice_profiles[profile_id]
            
            logger.info(f"Deleted voice profile: {profile.name} ({profile_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete profile {profile_id}: {e}")
            return False
    
    def _save_profile(self, profile: VoiceProfile):
        """Save profile to storage"""
        try:
            profile_file = self.profiles_dir / f"{profile.profile_id}.json"
            with open(profile_file, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save profile {profile.profile_id}: {e}")
    
    def _load_profiles(self):
        """Load all profiles from storage"""
        try:
            profile_files = list(self.profiles_dir.glob("*.json"))
            loaded_count = 0
            
            for profile_file in profile_files:
                try:
                    with open(profile_file, 'r') as f:
                        profile_data = json.load(f)
                    
                    profile = VoiceProfile.from_dict(profile_data)
                    
                    # Verify audio file exists
                    if os.path.exists(profile.reference_audio_path):
                        self.voice_profiles[profile.profile_id] = profile
                        loaded_count += 1
                    else:
                        logger.warning(f"Audio file missing for profile {profile.profile_id}")
                        
                except Exception as e:
                    logger.error(f"Failed to load profile from {profile_file}: {e}")
            
            logger.info(f"Loaded {loaded_count} voice profiles")
            
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")

class MultiSpeakerDialogueManager:
    """Manages multi-speaker dialogue generation"""
    
    def __init__(self, tts_model: FireRedTTS2, voice_manager: VoiceProfileManager):
        self.tts_model = tts_model
        self.voice_manager = voice_manager
        self.speaker_configs: Dict[str, MultiSpeakerConfig] = {}
    
    def create_dialogue_config(self, 
                             name: str,
                             description: str,
                             speakers: List[Dict[str, Any]],
                             dialogue_style: str = "conversational",
                             turn_management: str = "automatic",
                             voice_consistency: bool = True) -> str:
        """Create a multi-speaker dialogue configuration"""
        
        config_id = str(uuid.uuid4())
        
        config = MultiSpeakerConfig(
            config_id=config_id,
            name=name,
            description=description,
            speakers=speakers,
            dialogue_style=dialogue_style,
            turn_management=turn_management,
            voice_consistency=voice_consistency,
            created_at=datetime.now()
        )
        
        self.speaker_configs[config_id] = config
        return config_id
    
    def generate_multi_speaker_dialogue(self, 
                                      dialogue_text: str,
                                      config_id: Optional[str] = None,
                                      speaker_profiles: Optional[Dict[str, str]] = None) -> Tuple[bool, str, Optional[Tuple[int, np.ndarray]]]:
        """
        Generate multi-speaker dialogue audio
        
        Args:
            dialogue_text: Text with speaker tags like "[S1]Hello[S2]Hi there"
            config_id: Optional dialogue configuration ID
            speaker_profiles: Optional mapping of speaker tags to profile IDs
        
        Returns:
            (success, message, audio_output)
        """
        
        try:
            # Parse dialogue text into turns
            import re
            turns = re.findall(r'(\[S\d+\][^[]*)', dialogue_text)
            
            if not turns:
                return False, "No valid speaker turns found in dialogue text", None
            
            # Prepare speaker configurations
            if speaker_profiles:
                # Use voice cloning for specified speakers
                prompt_wav_list = []
                prompt_text_list = []
                
                for turn in turns:
                    speaker_tag = turn[:4]  # e.g., "[S1]"
                    
                    if speaker_tag in speaker_profiles:
                        profile_id = speaker_profiles[speaker_tag]
                        profile = self.voice_manager.get_profile(profile_id)
                        
                        if profile:
                            prompt_wav_list.append(profile.reference_audio_path)
                            prompt_text_list.append(profile.reference_text)
                            
                            # Update usage
                            self.voice_manager.update_profile_usage(profile_id)
                        else:
                            logger.warning(f"Profile not found: {profile_id}")
                            prompt_wav_list.append(None)
                            prompt_text_list.append(None)
                    else:
                        prompt_wav_list.append(None)
                        prompt_text_list.append(None)
                
                # Generate dialogue with voice cloning
                audio_tensor = self.tts_model.generate_dialogue(
                    text_list=turns,
                    prompt_wav_list=prompt_wav_list if any(prompt_wav_list) else None,
                    prompt_text_list=prompt_text_list if any(prompt_text_list) else None,
                    temperature=0.9,
                    topk=30
                )
            else:
                # Generate with random voices
                audio_tensor = self.tts_model.generate_dialogue(
                    text_list=turns,
                    temperature=0.9,
                    topk=30
                )
            
            # Convert to output format
            if isinstance(audio_tensor, torch.Tensor):
                audio_output = (24000, audio_tensor.squeeze().cpu().numpy())
            else:
                audio_output = (24000, audio_tensor)
            
            return True, f"Generated dialogue with {len(turns)} turns", audio_output
            
        except Exception as e:
            logger.error(f"Multi-speaker dialogue generation failed: {e}")
            return False, f"Generation failed: {e}", None

def create_voice_cloning_interface(tts_model: FireRedTTS2) -> Tuple[VoiceProfileManager, Dict[str, Any]]:
    """Create voice cloning interface components"""
    
    # Initialize managers
    voice_manager = VoiceProfileManager()
    dialogue_manager = MultiSpeakerDialogueManager(tts_model, voice_manager)
    
    def create_voice_profile(name, description, audio_data, reference_text, language, gender, age_range):
        """Create a new voice profile"""
        if not name.strip():
            return "❌ Profile name is required", None, ""
        
        if audio_data is None:
            return "❌ Reference audio is required", None, ""
        
        if not reference_text.strip():
            return "❌ Reference text is required", None, ""
        
        success, message, profile = voice_manager.create_voice_profile(
            name=name.strip(),
            description=description.strip(),
            audio_data=audio_data,
            reference_text=reference_text.strip(),
            language=language,
            gender=gender,
            age_range=age_range
        )
        
        if success:
            # Return updated profile list
            profiles = voice_manager.list_profiles()
            profile_choices = [(f"{p.name} ({p.language}, Quality: {p.quality_score:.2f})", p.profile_id) 
                             for p in profiles]
            return f"✅ {message}", gr.update(choices=profile_choices), ""
        else:
            return f"❌ {message}", None, ""
    
    def delete_voice_profile(profile_id):
        """Delete a voice profile"""
        if not profile_id:
            return "❌ No profile selected", None
        
        profile = voice_manager.get_profile(profile_id)
        if not profile:
            return "❌ Profile not found", None
        
        success = voice_manager.delete_profile(profile_id)
        
        if success:
            profiles = voice_manager.list_profiles()
            profile_choices = [(f"{p.name} ({p.language}, Quality: {p.quality_score:.2f})", p.profile_id) 
                             for p in profiles]
            return f"✅ Deleted profile '{profile.name}'", gr.update(choices=profile_choices)
        else:
            return "❌ Failed to delete profile", None
    
    def generate_with_voice_cloning(text, profile_id):
        """Generate speech with voice cloning"""
        if not text.strip():
            return "❌ Text is required", None
        
        if not profile_id:
            return "❌ No voice profile selected", None
        
        profile = voice_manager.get_profile(profile_id)
        if not profile:
            return "❌ Profile not found", None
        
        try:
            # Generate with voice cloning
            audio_tensor = tts_model.generate_monologue(
                text=text.strip(),
                prompt_wav=profile.reference_audio_path,
                prompt_text=profile.reference_text,
                temperature=0.9,
                topk=30
            )
            
            # Update usage
            voice_manager.update_profile_usage(profile_id)
            
            # Convert to output format
            if isinstance(audio_tensor, torch.Tensor):
                audio_output = (24000, audio_tensor.squeeze().cpu().numpy())
            else:
                audio_output = (24000, audio_tensor)
            
            return f"✅ Generated speech using '{profile.name}' voice", audio_output
            
        except Exception as e:
            logger.error(f"Voice cloning generation failed: {e}")
            return f"❌ Generation failed: {e}", None
    
    def generate_multi_speaker_dialogue(dialogue_text, speaker_mapping):
        """Generate multi-speaker dialogue"""
        if not dialogue_text.strip():
            return "❌ Dialogue text is required", None
        
        # Parse speaker mapping if provided
        speaker_profiles = {}
        if speaker_mapping:
            try:
                # Expected format: "S1:profile_id1,S2:profile_id2"
                for mapping in speaker_mapping.split(','):
                    if ':' in mapping:
                        speaker, profile_id = mapping.strip().split(':', 1)
                        speaker_profiles[f"[{speaker.strip()}]"] = profile_id.strip()
            except Exception as e:
                logger.warning(f"Failed to parse speaker mapping: {e}")
        
        success, message, audio_output = dialogue_manager.generate_multi_speaker_dialogue(
            dialogue_text=dialogue_text.strip(),
            speaker_profiles=speaker_profiles if speaker_profiles else None
        )
        
        if success:
            return f"✅ {message}", audio_output
        else:
            return f"❌ {message}", None
    
    def get_profile_info(profile_id):
        """Get detailed profile information"""
        if not profile_id:
            return "No profile selected"
        
        profile = voice_manager.get_profile(profile_id)
        if not profile:
            return "Profile not found"
        
        info = f"""
**Profile Information:**
- **Name:** {profile.name}
- **Description:** {profile.description}
- **Language:** {profile.language}
- **Gender:** {profile.gender}
- **Age Range:** {profile.age_range}
- **Quality Score:** {profile.quality_score:.2f}
- **Usage Count:** {profile.usage_count}
- **Created:** {profile.created_at.strftime('%Y-%m-%d %H:%M')}
- **Last Used:** {profile.last_used.strftime('%Y-%m-%d %H:%M')}

**Voice Characteristics:**
- **Dominant Frequency:** {profile.voice_characteristics.get('dominant_frequency_hz', 0):.1f} Hz
- **Spectral Centroid:** {profile.voice_characteristics.get('spectral_centroid_hz', 0):.1f} Hz
- **Dynamic Range:** {profile.voice_characteristics.get('dynamic_range', 0):.3f}
- **RMS Energy:** {profile.voice_characteristics.get('rms_energy', 0):.3f}

**Reference Text:**
"{profile.reference_text}"
        """
        
        return info
    
    # Get initial profile choices
    profiles = voice_manager.list_profiles()
    initial_profile_choices = [(f"{p.name} ({p.language}, Quality: {p.quality_score:.2f})", p.profile_id) 
                              for p in profiles]
    
    return voice_manager, {
        "create_profile": create_voice_profile,
        "delete_profile": delete_voice_profile,
        "generate_cloned": generate_with_voice_cloning,
        "generate_dialogue": generate_multi_speaker_dialogue,
        "get_profile_info": get_profile_info,
        "initial_choices": initial_profile_choices
    }