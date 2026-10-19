/*
 * Copyright 2024-2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.agentscope.core.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import io.agentscope.core.ReActAgent;
import io.agentscope.core.agent.test.MockModel;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.session.SessionHistoryMode;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.JsonFileAgentStateStore;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

class ReActAgentShutdownRecoveryTest {

    private static final RuntimeContext CONTEXT =
            RuntimeContext.builder().userId("alice").sessionId("interrupted-session").build();

    @TempDir Path directory;

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preservesNewInputAfterRestart(boolean streaming) {
        Msg oldInput = userMessage("old-id", "interrupted request");
        Msg newInput = userMessage("new-id", "new independent request");
        saveInterruptedState(List.of(oldInput));
        MockModel model = new MockModel("OK");

        try (ReActAgent agent = restoredAgent(model)) {
            execute(agent, List.of(newInput), streaming);

            assertEquals(
                    List.of("old-id", "new-id"),
                    model.getLastMessages().stream().map(Msg::getId).toList());
            assertFalse(agent.getAgentState(CONTEXT).isShutdownInterrupted());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preservesRepeatedTextWithANewMessageId(boolean streaming) {
        saveInterruptedState(List.of(userMessage("old-id", "please continue")));
        MockModel model = new MockModel("OK");

        try (ReActAgent agent = restoredAgent(model)) {
            execute(agent, List.of(userMessage("new-id", "please continue")), streaming);

            assertEquals(2, model.getLastMessages().size());
            assertEquals("new-id", model.getLastMessages().get(1).getId());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deduplicatesARetryWithThePersistedMessageId(boolean streaming) {
        saveInterruptedState(List.of(userMessage("request-id", "interrupted request")));
        MockModel model = new MockModel("OK");

        try (ReActAgent agent = restoredAgent(model)) {
            // A deserialized retry is a different object with the original message ID.
            execute(agent, List.of(userMessage("request-id", "interrupted request")), streaming);

            assertEquals(1, model.getLastMessages().size());
            assertEquals("request-id", model.getLastMessages().get(0).getId());
            assertFalse(agent.getAgentState(CONTEXT).isShutdownInterrupted());
        }
    }

    @Test
    void preservesNewMessagesInABatchContainingARetry() {
        saveInterruptedState(List.of(userMessage("old-id", "interrupted request")));
        MockModel model = new MockModel("OK");

        try (ReActAgent agent = restoredAgent(model)) {
            execute(
                    agent,
                    List.of(
                            userMessage("old-id", "interrupted request"),
                            userMessage("new-id", "additional instruction")),
                    false);

            assertEquals(
                    List.of("old-id", "new-id"),
                    model.getLastMessages().stream().map(Msg::getId).toList());
        }
    }

    @Test
    void preservesExternallyProvidedToolResultsAfterShutdown() {
        Msg pendingCall =
                Msg.builder()
                        .role(MsgRole.ASSISTANT)
                        .content(ToolUseBlock.builder().id("lookup-call").name("lookup").build())
                        .build();
        saveInterruptedState(
                List.of(userMessage("request-id", "interrupted request"), pendingCall));
        MockModel model = new MockModel("OK");

        try (ReActAgent agent = restoredAgent(model)) {
            Msg toolResult =
                    Msg.builder()
                            .role(MsgRole.TOOL)
                            .content(
                                    ToolResultBlock.of(
                                            "lookup-call",
                                            "lookup",
                                            TextBlock.builder().text("lookup answer").build()))
                            .build();
            execute(agent, List.of(toolResult), false);

            assertEquals(
                    List.of("lookup-call"),
                    model.getLastMessages().stream()
                            .flatMap(msg -> msg.getContentBlocks(ToolResultBlock.class).stream())
                            .map(ToolResultBlock::getId)
                            .toList());
            assertEquals(1, model.getCallCount());
        }
    }

    @Test
    void onlyDeduplicatesOnTheShutdownRecoveryCall() {
        Msg input = userMessage("request-id", "interrupted request");
        saveInterruptedState(List.of(input));
        MockModel model = new MockModel("OK");

        try (ReActAgent agent = restoredAgent(model)) {
            execute(agent, List.of(input), false);
            execute(agent, List.of(input), false);

            assertEquals(
                    2,
                    model.getLastMessages().stream()
                            .filter(msg -> "request-id".equals(msg.getId()))
                            .count());
        }
    }

    @ParameterizedTest
    @NullAndEmptySource
    @ValueSource(strings = {" "})
    void preservesMessagesWithoutAUsableId(String id) {
        saveInterruptedState(List.of(userMessage(id, "interrupted request")));
        MockModel model = new MockModel("OK");

        try (ReActAgent agent = restoredAgent(model)) {
            execute(agent, List.of(userMessage(id, "new independent request")), false);

            assertEquals(
                    List.of("interrupted request", "new independent request"),
                    model.getLastMessages().stream().map(Msg::getTextContent).toList());
        }
    }

    @Test
    void resumesWithoutInputAndPersistsTheClearedMarker() {
        saveInterruptedState(List.of(userMessage("old-id", "interrupted request")));
        MockModel model = new MockModel("OK");

        try (ReActAgent agent = restoredAgent(model)) {
            execute(agent, List.of(), false);

            assertEquals(1, model.getLastMessages().size());
        }

        AgentState restored =
                new JsonFileAgentStateStore(directory)
                        .get("alice", "interrupted-session", "agent_state", AgentState.class)
                        .orElseThrow();
        assertFalse(restored.isShutdownInterrupted());
    }

    private void saveInterruptedState(List<Msg> messages) {
        new JsonFileAgentStateStore(directory)
                .save(
                        "alice",
                        "interrupted-session",
                        "agent_state",
                        AgentState.builder()
                                .userId("alice")
                                .sessionId("interrupted-session")
                                .context(messages)
                                .shutdownInterrupted(true)
                                .build());
    }

    private ReActAgent restoredAgent(MockModel model) {
        return ReActAgent.builder()
                .name("recovery-agent")
                .model(model)
                .sessionHistoryMode(SessionHistoryMode.LEGACY)
                .stateStore(new JsonFileAgentStateStore(directory))
                .build();
    }

    private static Msg userMessage(String id, String text) {
        return Msg.builder().id(id).role(MsgRole.USER).textContent(text).build();
    }

    private static void execute(ReActAgent agent, List<Msg> input, boolean streaming) {
        if (streaming) {
            assertNotNull(
                    agent.stream(input, StreamOptions.defaults(), CONTEXT)
                            .blockLast(Duration.ofSeconds(5)));
        } else {
            assertNotNull(agent.call(input, CONTEXT).block(Duration.ofSeconds(5)));
        }
    }
}
