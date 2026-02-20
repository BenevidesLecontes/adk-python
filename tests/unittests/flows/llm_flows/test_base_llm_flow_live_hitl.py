# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from google.adk.agents.live_request_queue import LiveRequest
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.flows.llm_flows.base_llm_flow import _LIVE_PENDING_CONFIRMATION_STATE_KEY
from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
import pytest

from ... import testing_utils


class _TestBaseLlmFlow(BaseLlmFlow):
  pass


def _make_rc_function_call(
    original_fc: types.FunctionCall,
    rc_id: str = 'rc-001',
) -> types.FunctionCall:
  return types.FunctionCall(
      name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
      args={
          'originalFunctionCall': original_fc.model_dump(
              exclude_none=True, by_alias=True
          )
      },
      id=rc_id,
  )


def _make_confirmation_response(
    rc_id: str,
    confirmed: bool = True,
) -> types.Content:
  payload = ToolConfirmation(confirmed=confirmed).model_dump(by_alias=True)
  return types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                  id=rc_id,
                  response=payload,
              )
          )
      ],
  )


def _make_function_response_event(
    invocation_id: str, author: str
) -> Event:
  return Event(
      id=Event.new_id(),
      invocation_id=invocation_id,
      author=author,
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='mock_tool', id='orig-001', response={'result': 'ok'}
                  )
              )
          ],
      ),
  )


@pytest.mark.asyncio
async def test_capture_stores_confirmation_in_session_state():
  """Confirmation response data must be persisted to session.state."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  rc_id = 'rc-001'
  confirmation_content = _make_confirmation_response(rc_id, confirmed=True)

  flow = _TestBaseLlmFlow()
  flow._capture_live_confirmation_response(confirmation_content, ctx)

  state = ctx.session.state
  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY in state
  captured = state[_LIVE_PENDING_CONFIRMATION_STATE_KEY]
  assert captured['invocation_id'] == ctx.invocation_id
  assert rc_id in captured['confirmation_response_ids']
  assert rc_id in captured['confirmation_responses']


@pytest.mark.asyncio
async def test_capture_ignores_non_confirmation_function_responses():
  """Function responses for tools other than request_confirmation must be ignored."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  unrelated_content = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='some_other_tool',
                  id='id-999',
                  response={'result': 'whatever'},
              )
          )
      ],
  )

  flow = _TestBaseLlmFlow()
  flow._capture_live_confirmation_response(unrelated_content, ctx)

  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY not in ctx.session.state


@pytest.mark.asyncio
async def test_capture_ignores_response_without_id():
  """A confirmation function_response without an ID must be silently skipped."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  content = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                  id=None,  # missing ID
                  response={'confirmed': True},
              )
          )
      ],
  )

  flow = _TestBaseLlmFlow()
  flow._capture_live_confirmation_response(content, ctx)

  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY not in ctx.session.state


@pytest.mark.asyncio
async def test_dispatch_yields_nothing_when_no_pending_state():
  """Without pending state, _dispatch_live_confirmation must yield nothing."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  llm_request = LlmRequest()

  flow = _TestBaseLlmFlow()
  events = [e async for e in flow._dispatch_live_confirmation(ctx, llm_request)]

  assert events == []


@pytest.mark.asyncio
async def test_dispatch_yields_nothing_when_invocation_id_mismatch():
  """Stale state from a prior invocation must not trigger tool execution."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  ctx.session.state[_LIVE_PENDING_CONFIRMATION_STATE_KEY] = {
      'confirmation_response_ids': ['rc-001'],
      'confirmation_responses': {'rc-001': {'confirmed': True}},
      'invocation_id': 'different-invocation-id',
  }
  llm_request = LlmRequest()

  flow = _TestBaseLlmFlow()
  events = [e async for e in flow._dispatch_live_confirmation(ctx, llm_request)]

  assert events == []
  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY in ctx.session.state


@pytest.mark.asyncio
async def test_dispatch_executes_tool_and_clears_state():
  """Happy path: matching state + session event → tool executes, state cleared."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  original_fc = types.FunctionCall(
      name='transfer_money', args={'amount': 42}, id='orig-001'
  )
  rc_id = 'rc-001'
  rc_fc = _make_rc_function_call(original_fc, rc_id=rc_id)

  fc_event = Event(
      id=Event.new_id(),
      invocation_id=ctx.invocation_id,
      author='test_agent',
      content=types.Content(
          role='model', parts=[types.Part(function_call=rc_fc)]
      ),
  )
  ctx.session.events.append(fc_event)

  confirmation_payload = ToolConfirmation(confirmed=True).model_dump(
      by_alias=True
  )
  ctx.session.state[_LIVE_PENDING_CONFIRMATION_STATE_KEY] = {
      'confirmation_response_ids': [rc_id],
      'confirmation_responses': {rc_id: confirmation_payload},
      'invocation_id': ctx.invocation_id,
  }

  mock_response_event = _make_function_response_event(
      ctx.invocation_id, 'test_agent'
  )
  llm_request = LlmRequest()
  flow = _TestBaseLlmFlow()

  with mock.patch(
      'google.adk.flows.llm_flows.functions.handle_function_call_list_async',
      return_value=mock_response_event,
  ) as mock_handle:
    events = [
        e async for e in flow._dispatch_live_confirmation(ctx, llm_request)
    ]

  assert mock_response_event in events
  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY not in ctx.session.state
  mock_handle.assert_called_once()
  called_function_calls = mock_handle.call_args.args[1]
  assert len(called_function_calls) == 1
  assert called_function_calls[0].name == 'transfer_money'
  assert called_function_calls[0].id == 'orig-001'


@pytest.mark.asyncio
async def test_dispatch_skips_malformed_original_function_call():
  """Malformed originalFunctionCall payload must be skipped and state cleared."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  rc_id = 'rc-bad'
  bad_rc_fc = types.FunctionCall(
      name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
      args={'originalFunctionCall': 'not-a-dict'},
      id=rc_id,
  )
  fc_event = Event(
      id=Event.new_id(),
      invocation_id=ctx.invocation_id,
      author='test_agent',
      content=types.Content(
          role='model', parts=[types.Part(function_call=bad_rc_fc)]
      ),
  )
  ctx.session.events.append(fc_event)
  ctx.session.state[_LIVE_PENDING_CONFIRMATION_STATE_KEY] = {
      'confirmation_response_ids': [rc_id],
      'confirmation_responses': {rc_id: {'confirmed': True}},
      'invocation_id': ctx.invocation_id,
  }

  llm_request = LlmRequest()
  flow = _TestBaseLlmFlow()
  events = [e async for e in flow._dispatch_live_confirmation(ctx, llm_request)]

  assert events == []
  # State must be cleared so subsequent turn_completes are not misidentified.
  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY not in ctx.session.state


@pytest.mark.asyncio
async def test_dispatch_clears_state_when_no_session_events_match():
  """Pending state with no matching session events must be cleared, not retained.

  Without this, every subsequent control-only turn_complete in the same
  invocation would re-enter _dispatch_live_confirmation and be misclassified.
  """
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  ctx.session.state[_LIVE_PENDING_CONFIRMATION_STATE_KEY] = {
      'confirmation_response_ids': ['rc-ghost'],
      'confirmation_responses': {'rc-ghost': {'confirmed': True}},
      'invocation_id': ctx.invocation_id,
  }

  llm_request = LlmRequest()
  flow = _TestBaseLlmFlow()
  events = [e async for e in flow._dispatch_live_confirmation(ctx, llm_request)]

  assert events == []
  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY not in ctx.session.state


@pytest.mark.asyncio
async def test_postprocess_live_dispatches_hitl_on_control_turn_complete():
  """_postprocess_live must call _dispatch_live_confirmation on a control-only
  turn_complete (no content, no transcription, no error)."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  original_fc = types.FunctionCall(
      name='pay_bill', args={'amount': 99}, id='orig-pay'
  )
  rc_id = 'rc-pay'
  rc_fc = _make_rc_function_call(original_fc, rc_id=rc_id)

  fc_event = Event(
      id=Event.new_id(),
      invocation_id=ctx.invocation_id,
      author='test_agent',
      content=types.Content(
          role='model', parts=[types.Part(function_call=rc_fc)]
      ),
  )
  ctx.session.events.append(fc_event)

  confirmation_payload = ToolConfirmation(confirmed=True).model_dump(
      by_alias=True
  )
  ctx.session.state[_LIVE_PENDING_CONFIRMATION_STATE_KEY] = {
      'confirmation_response_ids': [rc_id],
      'confirmation_responses': {rc_id: confirmation_payload},
      'invocation_id': ctx.invocation_id,
  }

  control_turn_complete = LlmResponse(turn_complete=True)

  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=ctx.invocation_id,
      author='test_agent',
  )
  mock_response_event = _make_function_response_event(
      ctx.invocation_id, 'test_agent'
  )
  llm_request = LlmRequest()
  flow = _TestBaseLlmFlow()

  with mock.patch(
      'google.adk.flows.llm_flows.functions.handle_function_call_list_async',
      return_value=mock_response_event,
  ):
    events = [
        e
        async for e in flow._postprocess_live(
            ctx, llm_request, control_turn_complete, model_response_event
        )
    ]

  assert mock_response_event in events
  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY not in ctx.session.state


@pytest.mark.asyncio
async def test_postprocess_live_does_not_dispatch_when_turn_complete_has_content():
  """A turn_complete that also carries content must NOT trigger HITL dispatch."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )

  ctx.session.state[_LIVE_PENDING_CONFIRMATION_STATE_KEY] = {
      'confirmation_response_ids': ['rc-x'],
      'confirmation_responses': {'rc-x': {'confirmed': True}},
      'invocation_id': ctx.invocation_id,
  }

  llm_response_with_content = LlmResponse(
      turn_complete=True,
      content=types.Content(
          role='model',
          parts=[types.Part(text='Some spoken text.')],
      ),
  )
  model_response_event = Event(
      id=Event.new_id(),
      invocation_id=ctx.invocation_id,
      author='test_agent',
  )
  llm_request = LlmRequest()
  flow = _TestBaseLlmFlow()

  with mock.patch(
      'google.adk.flows.llm_flows.functions.handle_function_call_list_async'
  ) as mock_handle:
    [
        e
        async for e in flow._postprocess_live(
            ctx,
            llm_request,
            llm_response_with_content,
            model_response_event,
        )
    ]

  mock_handle.assert_not_called()
  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY in ctx.session.state


@pytest.mark.asyncio
async def test_send_to_model_captures_confirmation_response():
  """_send_to_model must call _capture_live_confirmation_response for
  confirmation function_responses before forwarding content to the model."""
  agent = LlmAgent(name='test_agent')
  ctx = await testing_utils.create_invocation_context(
      agent=agent, user_content=''
  )
  ctx.live_request_queue = LiveRequestQueue()

  rc_id = 'rc-send-001'
  confirmation_content = _make_confirmation_response(rc_id, confirmed=True)

  ctx.live_request_queue.send(LiveRequest(content=confirmation_content))
  ctx.live_request_queue.close()

  mock_connection = mock.AsyncMock()
  flow = _TestBaseLlmFlow()
  await flow._send_to_model(mock_connection, ctx)

  mock_connection.send_content.assert_called_once_with(confirmation_content)
  assert _LIVE_PENDING_CONFIRMATION_STATE_KEY in ctx.session.state
  captured = ctx.session.state[_LIVE_PENDING_CONFIRMATION_STATE_KEY]
  assert rc_id in captured['confirmation_response_ids']