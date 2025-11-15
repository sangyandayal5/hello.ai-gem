import { and, eq, not } from 'drizzle-orm'
import { NextRequest, NextResponse } from 'next/server'

import {
  CallTranscriptionReadyEvent,
  CallSessionParticipantLeftEvent,
  CallRecordingReadyEvent,
  CallSessionStartedEvent,
  CallEndedEvent,
} from '@stream-io/node-sdk'

import { db } from '@/db'
import { agents, meetings } from '@/db/schema'
import { streamVideo } from '@/lib/stream-video'
import { inngest } from '@/inngest/client'
import { geminiVoiceService } from '@/lib/gemini-voice'
import { generatedAvatarUri } from '@/lib/avatar'

interface PartialStreamPayload {
  call_cid?: string
  call?: { cid?: string }
  user?: { id?: string }
  closed_caption?: { user_id?: string }
  speaker_id?: string
}

function extractCaptionText(obj: unknown): string | null {
  try {
    if (!obj || typeof obj !== 'object') return null
    const o = obj as Record<string, unknown>
    // Direct known locations
    const cc = (o.closed_caption || o.caption) as
      | Record<string, unknown>
      | undefined
    if (cc) {
      const direct = cc.text || cc.caption || cc.content
      if (typeof direct === 'string' && direct.trim().length > 0) return direct
      // Some providers nest inside payload or data
      const payload = (cc.payload || cc.data) as
        | Record<string, unknown>
        | undefined
      if (payload) {
        const pText = payload.text || payload.caption || payload.content
        if (typeof pText === 'string' && pText.trim().length > 0) return pText
      }
    }
    // Fallback: shallow scan for stringy text-like fields
    for (const [k, v] of Object.entries(o)) {
      if (
        typeof v === 'string' &&
        ['text', 'caption', 'content'].includes(k) &&
        v.trim().length > 0
      ) {
        return v
      }
      if (v && typeof v === 'object') {
        const inner = extractCaptionText(v)
        if (inner) return inner
      }
    }
  } catch {}
  return null
}

function verifySignatureWithSDK(body: string, signature: string): boolean {
  return streamVideo.verifyWebhook(body, signature)
}

async function ensureGeminiSessionForMeeting(
  meetingId: string
): Promise<boolean> {
  try {
    if (geminiVoiceService.hasSession(meetingId)) return true

    const [meeting] = await db
      .select()
      .from(meetings)
      .where(eq(meetings.id, meetingId))
    if (!meeting) {
      console.warn(
        '[Webhook] ensureGeminiSession: meeting not found',
        meetingId
      )
      return false
    }

    const [agent] = await db
      .select()
      .from(agents)
      .where(eq(agents.id, meeting.agentId))
    if (!agent) {
      console.warn(
        '[Webhook] ensureGeminiSession: agent not found',
        meeting.agentId
      )
      return false
    }

    const call = streamVideo.video.call('default', meetingId)

    try {
      await streamVideo.upsertUsers([
        {
          id: agent.id,
          name: agent.name,
          role: 'user',
          image: generatedAvatarUri({
            seed: agent.name,
            variant: 'botttsNeutral',
          }),
        },
      ])
    } catch (e) {
      console.warn(
        '[Webhook] ensureGeminiSession: upsertUsers failed (continuing):',
        e
      )
    }

    await geminiVoiceService.startSession(call, agent.id, agent.instructions)
    console.log(
      '[Webhook] ensureGeminiSession: started Gemini session for',
      meetingId
    )
    return true
  } catch (e) {
    console.error('[Webhook] ensureGeminiSession error:', e)
    return false
  }
}

export async function POST(req: NextRequest) {
  const signature = req.headers.get('x-signature')
  const apiKey = req.headers.get('x-api-key')

  if (!signature || !apiKey) {
    return NextResponse.json(
      { error: 'Missing signature or API key' },
      { status: 400 }
    )
  }

  const body = await req.text()

  if (!verifySignatureWithSDK(body, signature)) {
    return NextResponse.json({ error: 'Invalid signature' }, { status: 401 })
  }

  let payload: unknown
  try {
    payload = JSON.parse(body) as Record<string, unknown>
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 })
  }

  const eventType = (payload as Record<string, unknown>)?.type
  console.log('[Webhook] Incoming event:', eventType)

  if (eventType === 'call.session_started') {
    const event = payload as CallSessionStartedEvent
    const meetingId = event.call.custom?.meetingId

    if (!meetingId) {
      return NextResponse.json({ error: 'Missing meetingId' }, { status: 400 })
    }

    const [existingMeeting] = await db
      .select()
      .from(meetings)
      .where(
        and(
          eq(meetings.id, meetingId),
          not(eq(meetings.status, 'active')),
          not(eq(meetings.status, 'completed')),
          not(eq(meetings.status, 'cancelled')),
          not(eq(meetings.status, 'processing'))
        )
      )

    if (!existingMeeting) {
      return NextResponse.json({ error: 'Meeting not found' }, { status: 404 })
    }

    await db
      .update(meetings)
      .set({
        status: 'active',
        startedAt: new Date(),
      })
      .where(eq(meetings.id, existingMeeting.id))

    const [existingAgent] = await db
      .select()
      .from(agents)
      .where(eq(agents.id, existingMeeting.agentId))

    if (!existingAgent) {
      return NextResponse.json({ error: 'Agent not found' }, { status: 404 })
    }

    const call = streamVideo.video.call('default', meetingId)
    try {
      const doUpsert = async () =>
        await streamVideo.upsertUsers([
          {
            id: existingAgent.id,
            name: existingAgent.name,
            role: 'user',
            image: generatedAvatarUri({
              seed: existingAgent.name,
              variant: 'botttsNeutral',
            }),
          },
        ])

      try {
        await doUpsert()
      } catch (e1) {
        console.warn('[Agent] upsertUsers retry after error:', e1)
        await new Promise((r) => setTimeout(r, 500))
        await doUpsert()
      }

      // Join agent visibly via OpenAI realtime (silent) if key present
      if (process.env.OPENAI_API_KEY) {
        try {
          const realtimeClient = await streamVideo.video.connectOpenAi({
            call,
            openAiApiKey: process.env.OPENAI_API_KEY!,
            agentUserId: existingAgent.id,
          })

          // Make sure OpenAI agent stays silent; Gemini handles answers
          await realtimeClient.updateSession({
            instructions:
              'You are a silent listener. Do not respond or speak. Gemini will provide responses.',
          })

        } catch (err) {
          console.error(
            '[Agent] OpenAI join failed (continuing without it):',
            err
          )
        }
      }

      // Start Gemini session (actual brain + optional TTS)
      await geminiVoiceService.startSession(
        call,
        existingAgent.id,
        existingAgent.instructions
      )

      console.log(`[Gemini Voice] Started session for meeting ${meetingId}`)
      console.log(
        `[Gemini Voice] Agent ${existingAgent.id} ready to process transcriptions`
      )
    } catch (error) {
      console.error('FAILED TO CONNECT GEMINI VOICE AGENT:', error)
    }
  } else if (eventType === 'call.session_participant_left') {
    const event = payload as CallSessionParticipantLeftEvent
    const meetingId = event.call_cid.split(':')[1]

    if (!meetingId) {
      return NextResponse.json({ error: 'Missing meetingId' }, { status: 400 })
    }

    // Do not end session; wait for full call.session_ended
    console.log(
      '[Webhook] participant_left; keeping session alive for',
      meetingId
    )
  } else if (eventType === 'call.session_ended') {
    const event = payload as CallEndedEvent
    const meetingId = event.call.custom?.meetingId

    if (!meetingId) {
      return NextResponse.json({ error: 'Missing meetingId' }, { status: 400 })
    }

    // End Gemini voice session
    await geminiVoiceService.endSession(meetingId)

    await db
      .update(meetings)
      .set({
        status: 'processing',
        endedAt: new Date(),
      })
      .where(and(eq(meetings.id, meetingId), eq(meetings.status, 'active')))
  } else if (eventType === 'call.transcription_ready') {
    const event = payload as CallTranscriptionReadyEvent
    const meetingId = event.call_cid.split(':')[1]

    const [updatedMeeting] = await db
      .update(meetings)
      .set({
        transcriptUrl: event.call_transcription.url,
      })
      .where(eq(meetings.id, meetingId))
      .returning()

    if (!updatedMeeting) {
      return NextResponse.json({ error: 'Meeting not found' }, { status: 404 })
    }

    const hasSession = geminiVoiceService.hasSession(meetingId)
    console.log('[Webhook] transcription_ready; has session?', hasSession)
    // Process transcription with Gemini if session exists
    // Stream Video sends transcription data - we need to fetch and process it
    if (hasSession) {
      try {
        // Fetch transcript data
        const transcriptResponse = await fetch(event.call_transcription.url)
        console.log(
          '[Webhook] transcript fetch status:',
          transcriptResponse.status
        )
        const transcriptData = await transcriptResponse.text()
        console.log('[Webhook] transcript bytes:', transcriptData.length)

        // Parse transcript (assuming JSONL format)
        const transcriptLines = transcriptData.split('\n').filter(Boolean)
        console.log('[Webhook] transcript lines:', transcriptLines.length)
        for (const line of transcriptLines) {
          try {
            const item = JSON.parse(line)
            if (item.text && item.speaker_id) {
              console.log(
                '[Webhook] forwarding to Gemini:',
                (item.text as string).slice(0, 80)
              )
              await geminiVoiceService.processTranscription(
                meetingId,
                item.text,
                item.speaker_id
              )
            }
          } catch (parseError) {
            console.error('Error parsing transcript line:', parseError)
          }
        }
      } catch (error) {
        console.error('Error processing transcription with Gemini:', error)
      }
    }

    await inngest.send({
      name: 'meetings/processing',
      data: {
        meetingId: updatedMeeting.id,
        transcriptUrl: updatedMeeting.transcriptUrl,
      },
    })
  } else if (eventType === 'call.recording_ready') {
    const event = payload as CallRecordingReadyEvent
    const meetingId = event.call_cid.split(':')[1]

    await db
      .update(meetings)
      .set({
        recordingUrl: event.call_recording.url,
      })
      .where(eq(meetings.id, meetingId))
  } else if (eventType === 'call.closed_caption') {
    // Live captions: forward text to Gemini
    try {
      const anyPayload = payload as PartialStreamPayload
      const meetingId: string | undefined =
        anyPayload.call_cid?.split(':')[1] ||
        anyPayload.call?.cid?.split(':')[1]

      if (!meetingId) {
        console.warn('[Webhook] closed_caption missing meetingId')
        return NextResponse.json({ status: 'ignored' })
      }

      // Ensure session exists (lazy create)
      const ready = await ensureGeminiSessionForMeeting(meetingId)
      if (!ready) {
        console.warn(
          '[Webhook] closed_caption could not ensure session for',
          meetingId
        )
        return NextResponse.json({ status: 'ignored' })
      }

      const text: string | null = extractCaptionText(anyPayload)
      const speakerId: string =
        anyPayload.user?.id ||
        anyPayload.closed_caption?.user_id ||
        anyPayload.speaker_id ||
        'user'

      if (!text || text.trim().length === 0) {
        console.log(
          '[Webhook] closed_caption empty text, payload keys:',
          Object.keys(anyPayload || {})
        )
        return NextResponse.json({ status: 'ignored' })
      }

      console.log('[Webhook] closed_caption → Gemini:', text.slice(0, 120))
      await geminiVoiceService.processTranscription(meetingId, text, speakerId)
    } catch (err) {
      console.error('[Webhook] closed_caption handler error:', err)
    }
  } else if (
    eventType === 'call.closed_captions_started' ||
    eventType === 'call.transcription_started'
  ) {
    try {
      const anyPayload = payload as PartialStreamPayload
      const meetingId: string | undefined =
        anyPayload.call_cid?.split(':')[1] ||
        anyPayload.call?.cid?.split(':')[1]
      if (!meetingId) return NextResponse.json({ status: 'ignored' })
      const ready = await ensureGeminiSessionForMeeting(meetingId)
      console.log(
        '[Webhook] ensure session on start event:',
        eventType,
        meetingId,
        'ready?',
        ready
      )
    } catch (e) {
      console.error('[Webhook] start event ensure session error:', e)
    }
  }

  return NextResponse.json({ status: 'ok' })
}
