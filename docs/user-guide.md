# FireRedTTS2 User Guide

## Overview

This guide explains how to use the FireRedTTS2 web interface for text-to-speech synthesis, voice cloning, multi-speaker dialogue generation, and real-time speech-to-speech conversations. The interface is designed to be intuitive and accessible through any modern web browser.

## Getting Started

### Accessing the Interface

1. **Open Your Browser**
   - Use Chrome, Firefox, Safari, or Edge (latest versions recommended)
   - Ensure JavaScript is enabled
   - Allow microphone access when prompted

2. **Navigate to the Application**
   - Enter the RunPod URL provided by your administrator
   - Example: `https://abc123-7860.proxy.runpod.net`
   - Wait for the interface to load completely

3. **Initial Setup**
   - Grant microphone permissions when requested
   - Test audio output by adjusting your system volume
   - Verify the interface loads without errors

## Interface Overview

### Main Navigation Tabs

The interface is organized into several main sections:

1. **Text-to-Speech**: Basic TTS functionality
2. **Voice Cloning**: Create and use custom voices
3. **Multi-Speaker Dialogue**: Generate conversations between multiple speakers
4. **Speech-to-Speech**: Real-time conversation mode
5. **Performance Monitor**: System metrics and diagnostics
6. **Settings**: Configuration and preferences

## Text-to-Speech Features

### Basic Text-to-Speech

1. **Enter Text**
   - Type or paste text in the input field (up to 1000 characters)
   - Supports multiple languages and special characters
   - Use punctuation for natural speech rhythm

2. **Select Voice**
   - Choose from available pre-trained voices
   - Preview voices with sample text
   - Adjust voice parameters if available

3. **Generate Speech**
   - Click "Generate" to create audio
   - Wait for processing (typically 2-5 seconds)
   - Audio will play automatically when ready

4. **Download Audio**
   - Click "Download" to save audio file
   - Available formats: WAV, MP3
   - Files are automatically named with timestamp

### Advanced TTS Options

**Voice Parameters**
- **Speed**: Adjust speaking rate (0.5x to 2.0x)
- **Pitch**: Modify voice pitch (-12 to +12 semitones)
- **Volume**: Control output volume (0% to 150%)
- **Emotion**: Select emotional tone (if supported)

**Text Processing Options**
- **Language**: Auto-detect or manually select
- **SSML Support**: Use Speech Synthesis Markup Language tags
- **Pronunciation**: Add custom pronunciations
- **Pauses**: Insert custom pauses with `<pause>` tags

## Voice Cloning

### Creating a Voice Clone

1. **Prepare Reference Audio**
   - Record 10-30 seconds of clear speech
   - Use quiet environment with minimal background noise
   - Speak naturally at normal pace
   - Supported formats: WAV, MP3, FLAC, WebM

2. **Upload Reference Audio**
   - Click "Upload Reference Audio" button
   - Select your audio file (max 50MB)
   - Wait for upload and processing

3. **Provide Reference Text**
   - Enter the exact text spoken in the reference audio
   - Ensure accuracy for best cloning results
   - Use proper punctuation and capitalization

4. **Create Voice Profile**
   - Enter a name for your voice clone
   - Select language (auto-detected from audio)
   - Click "Create Voice Clone"
   - Wait for processing (1-3 minutes)

### Using Voice Clones

1. **Select Cloned Voice**
   - Navigate to voice selection dropdown
   - Choose from "Custom Voices" section
   - Preview with sample text

2. **Generate with Cloned Voice**
   - Enter your text
   - Select the cloned voice
   - Generate audio as normal

3. **Manage Voice Clones**
   - View all created voices in "Voice Management"
   - Delete unwanted voices
   - Rename voice profiles
   - Export/import voice profiles

### Voice Cloning Tips

**For Best Results:**
- Use high-quality audio (22kHz or higher sample rate)
- Ensure consistent volume levels
- Avoid background noise and echo
- Speak clearly and naturally
- Use emotionally neutral tone for reference

**Common Issues:**
- Robotic sound: Try longer reference audio
- Wrong accent: Ensure reference matches target language
- Inconsistent quality: Check audio quality and noise levels

## Multi-Speaker Dialogue

### Creating Dialogue Scenes

1. **Set Up Speakers**
   - Click "Add Speaker" for each character
   - Assign names and select voices
   - Choose from pre-trained or cloned voices

2. **Write Dialogue Script**
   - Use format: `Speaker Name: Dialogue text`
   - Example:
     ```
     Alice: Hello, how are you today?
     Bob: I'm doing great, thanks for asking!
     Alice: That's wonderful to hear.
     ```

3. **Configure Scene Settings**
   - **Pause Between Speakers**: Adjust timing (0.5-3.0 seconds)
   - **Background Audio**: Add ambient sounds (optional)
   - **Scene Length**: Set maximum duration
   - **Export Format**: Choose audio format

4. **Generate Dialogue**
   - Click "Generate Dialogue"
   - Monitor progress in real-time
   - Preview individual speaker lines
   - Download complete scene

### Advanced Dialogue Features

**Speaker Emotions**
- Add emotion tags: `Alice [happy]: Great news everyone!`
- Available emotions: happy, sad, angry, excited, calm, surprised

**Scene Directions**
- Add pauses: `[pause 2s]`
- Sound effects: `[sfx: door closing]`
- Background changes: `[bg: restaurant ambiance]`

**Conversation Flow**
- Automatic turn-taking based on punctuation
- Interrupt detection and handling
- Natural conversation pacing

## Speech-to-Speech Conversation

### Starting a Conversation

1. **Enable Conversation Mode**
   - Click "Speech-to-Speech" tab
   - Select conversation settings
   - Choose your voice for responses

2. **Configure Conversation**
   - **Response Voice**: Select TTS voice for AI responses
   - **Language**: Set conversation language
   - **Personality**: Choose AI personality (friendly, professional, casual)
   - **Response Length**: Set preferred response length

3. **Begin Conversation**
   - Click "Start Conversation"
   - Speak when the microphone indicator is active
   - Wait for AI response
   - Continue natural conversation flow

### Conversation Controls

**During Conversation:**
- **Mute/Unmute**: Control microphone input
- **Pause**: Temporarily pause conversation
- **Stop**: End conversation session
- **Volume**: Adjust response volume
- **Speed**: Change response speaking rate

**Conversation History:**
- View complete conversation transcript
- Replay any part of the conversation
- Export conversation as text or audio
- Save conversation for later review

### Voice Activity Detection

The system automatically detects when you're speaking:

- **Green Indicator**: Currently speaking
- **Yellow Indicator**: Processing your speech
- **Blue Indicator**: AI is responding
- **Gray Indicator**: Waiting for input

**Adjusting Sensitivity:**
- **High Sensitivity**: Picks up quiet speech, may trigger on noise
- **Medium Sensitivity**: Balanced for normal environments
- **Low Sensitivity**: Requires clear, loud speech

## Performance Monitoring

### Real-Time Metrics

The performance monitor shows:

**Processing Times:**
- **ASR Latency**: Speech recognition time
- **LLM Latency**: AI response generation time
- **TTS Latency**: Speech synthesis time
- **Total Latency**: End-to-end processing time

**System Resources:**
- **GPU Utilization**: Graphics card usage
- **Memory Usage**: RAM and VRAM consumption
- **Active Sessions**: Current user count
- **Queue Status**: Pending requests

**Audio Quality:**
- **Sample Rate**: Audio quality indicator
- **Bit Depth**: Audio resolution
- **Compression**: Audio compression ratio
- **Noise Level**: Background noise detection

### Performance Optimization

**For Better Performance:**
- Use shorter text inputs when possible
- Avoid multiple simultaneous requests
- Close unused browser tabs
- Use wired internet connection for stability

**Quality vs Speed Trade-offs:**
- **High Quality**: Better audio, slower processing
- **Balanced**: Good quality, reasonable speed
- **Fast**: Lower quality, faster processing

## Settings and Configuration

### Audio Settings

**Input Settings:**
- **Microphone**: Select input device
- **Input Volume**: Adjust microphone sensitivity
- **Noise Suppression**: Enable/disable noise reduction
- **Echo Cancellation**: Reduce audio feedback

**Output Settings:**
- **Speakers**: Select output device
- **Output Volume**: Master volume control
- **Audio Format**: Choose quality/compatibility
- **Buffer Size**: Adjust for latency/stability

### Interface Settings

**Display Options:**
- **Theme**: Light/dark mode
- **Language**: Interface language
- **Font Size**: Adjust text size
- **Animations**: Enable/disable UI animations

**Behavior Settings:**
- **Auto-play**: Automatically play generated audio
- **Auto-save**: Save generated audio automatically
- **Notifications**: Enable system notifications
- **Keyboard Shortcuts**: Enable hotkeys

### Privacy Settings

**Data Handling:**
- **Save Conversations**: Keep conversation history
- **Voice Profile Storage**: Store custom voices
- **Usage Analytics**: Share anonymous usage data
- **Audio Caching**: Cache audio for faster playback

## Troubleshooting

### Common Issues

**Audio Not Playing:**
1. Check browser audio permissions
2. Verify system volume settings
3. Try different audio format
4. Refresh the page and try again

**Microphone Not Working:**
1. Grant microphone permissions
2. Check browser microphone settings
3. Test microphone in other applications
4. Try different browser

**Poor Audio Quality:**
1. Check internet connection stability
2. Reduce background noise
3. Use higher quality settings
4. Try different voice model

**Slow Performance:**
1. Close other browser tabs
2. Check system resources
3. Use shorter text inputs
4. Try during off-peak hours

### Getting Help

**Built-in Help:**
- Hover over any interface element for tooltips
- Click "?" icons for contextual help
- Check the FAQ section in settings

**Error Messages:**
- Read error messages carefully
- Try suggested solutions
- Refresh page if errors persist
- Contact administrator for persistent issues

## Keyboard Shortcuts

### Global Shortcuts
- **Ctrl+Enter**: Generate/Submit
- **Ctrl+S**: Save current audio
- **Ctrl+R**: Refresh interface
- **Escape**: Cancel current operation

### Conversation Mode
- **Space**: Push-to-talk (hold to speak)
- **M**: Mute/unmute microphone
- **P**: Pause/resume conversation
- **S**: Stop conversation

### Audio Playback
- **Space**: Play/pause audio
- **Left Arrow**: Rewind 5 seconds
- **Right Arrow**: Forward 5 seconds
- **Up Arrow**: Increase volume
- **Down Arrow**: Decrease volume

## Best Practices

### For Optimal Results

**Text Input:**
- Use clear, well-punctuated text
- Avoid excessive capitalization
- Include natural pauses with commas and periods
- Test with shorter texts first

**Voice Cloning:**
- Use high-quality reference audio
- Ensure quiet recording environment
- Speak naturally and clearly
- Provide accurate reference text

**Conversations:**
- Speak clearly and at normal pace
- Wait for response before speaking again
- Use good microphone and quiet environment
- Keep responses conversational

### Resource Management

**Efficient Usage:**
- Generate shorter audio clips when testing
- Close unused browser tabs
- Log out when finished
- Clear browser cache periodically

**Quality Optimization:**
- Use appropriate quality settings for your needs
- Test different voices to find best fit
- Adjust settings based on your hardware
- Monitor performance metrics

## Advanced Features

### API Integration

For developers, the system provides REST API access:

**Endpoints:**
- `/api/tts`: Text-to-speech generation
- `/api/clone`: Voice cloning operations
- `/api/dialogue`: Multi-speaker dialogue
- `/api/conversation`: Speech-to-speech conversation

**WebSocket API:**
- Real-time audio streaming
- Live conversation updates
- Performance monitoring
- System notifications

### Custom Integrations

**Webhook Support:**
- Receive notifications on completion
- Integrate with external systems
- Automate workflows
- Custom event handling

**Batch Processing:**
- Process multiple texts simultaneously
- Bulk voice cloning operations
- Automated dialogue generation
- Scheduled processing

## Support and Resources

### Documentation
- **API Documentation**: Available at `/docs` endpoint
- **Developer Guide**: Technical implementation details
- **Admin Guide**: System administration and maintenance

### Community
- **User Forum**: Share tips and get help from other users
- **Feature Requests**: Suggest new features and improvements
- **Bug Reports**: Report issues and track fixes

### Contact
- **Technical Support**: Contact your system administrator
- **Feature Questions**: Refer to this user guide
- **Bug Reports**: Use the built-in feedback system

---

This user guide covers all major features of the FireRedTTS2 system. For technical details and advanced configuration, refer to the administrator documentation.