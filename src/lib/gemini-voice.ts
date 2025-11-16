import 'server-only'

import { GoogleGenerativeAI } from '@google/generative-ai'
import * as tts from '@google-cloud/text-to-speech'
import { StreamVideoClient } from '@stream-io/node-sdk'

// 1. Import Cloudinary and Stream
import { v2 as cloudinary } from 'cloudinary'
import { Readable } from 'stream'

// 2. Configure Cloudinary
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
})

type Call = ReturnType<StreamVideoClient['call']>

// Initialize clients
let ttsClient: tts.TextToSpeechClient | null = null
let geminiClient: GoogleGenerativeAI | null = null

// --- NEW: Robust Credential Helper ---
function getGoogleCredentials() {
  // Priority 1: Separate Env Vars (Best for Vercel)
  if (process.env.GOOGLE_CLIENT_EMAIL && process.env.GOOGLE_PRIVATE_KEY) {
    return {
      client_email: process.env.GOOGLE_CLIENT_EMAIL,
      // Fix: Replace escaped newlines with real newlines
      private_key: process.env.GOOGLE_PRIVATE_KEY.replace(/\\n/g, '\n'),
      project_id: process.env.GOOGLE_PROJECT_ID,
    }
  }

  // Priority 2: Base64 Env Var (Legacy/Backup)
  if (process.env.GOOGLE_CLOUD_CREDENTIALS_BASE64) {
    try {
      const decoded = Buffer.from(
        process.env.GOOGLE_CLOUD_CREDENTIALS_BASE64,
        'base64'
      ).toString('utf-8')
      return JSON.parse(decoded)
    } catch (error) {
      console.error('[Gemini Voice] Failed to parse Google credentials:', error)
      return null
    }
  }
  return null
}
// --------------------------------------------------------------------

function getTTSClient() {
  if (!ttsClient) {
    // 1. Try getting credentials from Env Vars (Production)
    const credentials = getGoogleCredentials()

    if (credentials) {
      ttsClient = new tts.TextToSpeechClient({
        credentials: {
          client_email: credentials.client_email,
          private_key: credentials.private_key,
        },
        projectId: credentials.project_id,
      })
    }
    // 2. Fallback: Try getting credentials from Local File (Local Dev)
    else if (process.env.GOOGLE_CLOUD_KEYFILE) {
      // We only use this if the file path actually exists in env
      ttsClient = new tts.TextToSpeechClient({
        keyFilename: process.env.GOOGLE_CLOUD_KEYFILE,
      })
    }
  }
  return ttsClient
}

function getGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_GENAI_API_KEY
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY or GOOGLE_GENAI_API_KEY is required')
  }
  if (!geminiClient) {
    geminiClient = new GoogleGenerativeAI(apiKey)
  }
  return geminiClient
}

interface GeminiVoiceSession {
  call: Call
  agentUserId: string
  instructions: string
  conversationHistory: Array<{ role: string; content: string }>
  isProcessing: boolean
  audioResponses: Array<{ text: string; audioUrl: string; timestamp: Date }>
}

export class GeminiVoiceService {
  private sessions: Map<string, GeminiVoiceSession> = new Map()

  async startSession(
    call: Call,
    agentUserId: string,
    instructions: string
  ): Promise<void> {
    const sessionId = call.id

    this.sessions.set(sessionId, {
      call,
      agentUserId,
      instructions,
      conversationHistory: [],
      isProcessing: false,
      audioResponses: [],
    })
  }

  async processTranscription(
    callId: string,
    transcriptionText: string,
    speakerId: string
  ): Promise<void> {
    const session = this.sessions.get(callId)
    if (!session || session.isProcessing) {
      return
    }

    if (speakerId === session.agentUserId) {
      return
    }

    session.isProcessing = true

    try {
      session.conversationHistory.push({
        role: 'user',
        content: transcriptionText,
      })

      const geminiResponse = await this.generateGeminiResponse(
        session.instructions,
        session.conversationHistory
      )

      session.conversationHistory.push({
        role: 'assistant',
        content: geminiResponse,
      })

      console.log(
        '[Gemini Voice] TTS request text:',
        geminiResponse.slice(0, 120)
      )
      const audioBuffer = await this.textToSpeech(geminiResponse)
      console.log(
        '[Gemini Voice] TTS response buffer bytes:',
        audioBuffer ? audioBuffer.length : 0
      )

      if (audioBuffer) {
        try {
          // 3. UPLOAD TO CLOUDINARY
          console.log('[Gemini Voice] Uploading to Cloudinary...')
          const audioUrl = await this.uploadToCloudinary(audioBuffer, callId)

          session.audioResponses.push({
            text: geminiResponse,
            audioUrl, // This is now a remote URL (https://res.cloudinary.com/...)
            timestamp: new Date(),
          })

          console.log(
            `[Gemini Voice] Uploaded audio to Cloudinary: ${audioUrl}`
          )
        } catch (saveError) {
          console.error(
            '[Gemini Voice] Error uploading audio to Cloudinary:',
            saveError
          )
        }
      } else {
        try {
          session.audioResponses.push({
            text: geminiResponse,
            audioUrl: '',
            timestamp: new Date(),
          })
        } catch (error) {
          console.error('[Gemini Voice] Error storing text response:', error)
        }
      }

      console.log(
        `[Gemini Voice] Processed transcription for call ${callId}: ${transcriptionText.substring(
          0,
          50
        )}...`
      )
    } catch (error) {
      console.error('[Gemini Voice] Error processing transcription:', error)
    } finally {
      session.isProcessing = false
    }
  }

  // 4. Helper function to handle the Stream upload
  private async uploadToCloudinary(
    buffer: Buffer,
    callId: string
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      const uploadStream = cloudinary.uploader.upload_stream(
        {
          resource_type: 'auto', // Important: Let Cloudinary detect it's audio
          public_id: `voice_agent/${callId}-${Date.now()}`, // Organize in folder
          format: 'wav',
        },
        (error, result) => {
          if (error) {
            console.error('Cloudinary upload error details:', error)
            return reject(error)
          }
          if (result?.secure_url) {
            resolve(result.secure_url)
          } else {
            reject(new Error('Cloudinary upload failed to return URL'))
          }
        }
      )

      // Convert Buffer to Stream and pipe to Cloudinary
      Readable.from(buffer).pipe(uploadStream)
    })
  }

  private async generateGeminiResponse(
    instructions: string,
    history: Array<{ role: string; content: string }>
  ): Promise<string> {
    const genAI = getGeminiClient()
    const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash-exp' })

    const systemPrompt = `${instructions}\n\nYou are having a conversation. Respond naturally and concisely.`

    const conversationParts = history.map((msg) => {
      return `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`
    })

    const prompt = `${systemPrompt}\n\nConversation:\n${conversationParts.join(
      '\n\n'
    )}\n\nAssistant:`

    console.log('[Gemini Voice] Generating response with Gemini...')
    const result = await model.generateContent(prompt)
    const response = await result.response
    const text = response.text()

    return text || 'I apologize, but I could not generate a response.'
  }

  private async textToSpeech(text: string): Promise<Buffer | null> {
    const client = getTTSClient()
    if (!client) {
      return null
    }

    const request: tts.protos.google.cloud.texttospeech.v1.ISynthesizeSpeechRequest =
      {
        input: { text },
        voice: {
          languageCode: 'en-US',
          name: 'en-US-Neural2-F',
          ssmlGender: 'FEMALE' as const,
        },
        audioConfig: {
          audioEncoding: 'LINEAR16' as const,
          sampleRateHertz: 16000,
        },
      }

    const [response] = await client.synthesizeSpeech(request)

    if (!response.audioContent) {
      throw new Error('Failed to generate audio from text')
    }

    return Buffer.from(response.audioContent)
  }

  getLatestAudioResponse(callId: string): string | null {
    const session = this.sessions.get(callId)
    if (!session || session.audioResponses.length === 0) {
      return null
    }
    const latest = session.audioResponses[session.audioResponses.length - 1]
    return latest.audioUrl
  }

  getAudioResponses(
    callId: string
  ): Array<{ text: string; audioUrl: string; timestamp: Date }> {
    const session = this.sessions.get(callId)
    return session?.audioResponses || []
  }

  async endSession(callId: string): Promise<void> {
    this.sessions.delete(callId)
    console.log(`[Gemini Voice] Session ended for call ${callId}`)
  }

  hasSession(callId: string): boolean {
    return this.sessions.has(callId)
  }
}

export const geminiVoiceService = new GeminiVoiceService()
