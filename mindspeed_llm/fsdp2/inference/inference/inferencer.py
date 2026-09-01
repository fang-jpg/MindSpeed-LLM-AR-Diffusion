import torch.distributed as dist

from mindspeed_llm.fsdp2.utils.logging import get_logger
from .chat_model import ChatModel


logger = get_logger(__name__)


class Inferencer:
    """
    Executes inference tasks, backed by an extensible ChatModel engine.
    """
    def __init__(self, model, tokenizer, args):
        self.chat_model = ChatModel(model, tokenizer, args)
        self.args = args
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def run_interactive_chat(self):
        """
        Starts an interactive chat loop (CLI Demo) with streaming support.
        """
        logger.info_rank0(">>> Entering Interactive Chat Mode. Type 'exit' to quit.")
        
        history = []
        
        # --- Loop Invariants ---
        # Extracted outside the while loop to avoid repeated memory allocation
        EXIT_COMMANDS = ("exit", "quit")
        ROLE_USER, ROLE_ASSISTANT = ("user", "assistant")
        use_chat_template = getattr(self.args, "use_chat_template", True)
        base_chat = getattr(self.args, "base_chat", False)
        OUTPUT_PREFIX = "Assistant: " if (use_chat_template or base_chat) else "Completion: "
        VISUAL_SEPARATOR = "\n" + "-" * 40

        while True:
            # 1. Sync input across all FSDP ranks to prevent deadlocks
            user_input = self._get_sync_input()

            if user_input.lower() in EXIT_COMMANDS:
                break
            
            if not user_input:
                continue

            if base_chat:
                history.append({"role": ROLE_USER, "content": user_input})
                inference_messages = [{
                    "role": ROLE_USER,
                    "content": self._build_base_chat_prompt(history),
                }]
            elif use_chat_template:
                history.append({"role": ROLE_USER, "content": user_input})
                inference_messages = history
            else:
                # Base models are text-completion models, not multi-turn chat
                # models. Do not feed previous assistant completions back as a
                # role-formatted conversation.
                inference_messages = [{"role": ROLE_USER, "content": user_input}]
            
            if self.rank == 0:
                print(OUTPUT_PREFIX, end="", flush=True)

            if base_chat:
                # Use the non-streaming API so generated role markers can be
                # removed before displaying and storing the response.
                result = self.chat_model.chat(
                    inference_messages,
                    use_chat_template=False,
                )
                response = self._trim_base_chat_response(result[0].response_text)
                if self.rank == 0:
                    print(response, end="", flush=True)
            else:
                response = ""

                # 2. Use the streaming API for the typewriter effect
                for new_text in self.chat_model.stream_chat(
                    inference_messages,
                    use_chat_template=use_chat_template,
                ):
                    if self.rank == 0:
                        print(new_text, end="", flush=True)
                    response += new_text

            # Print a visual separator after the response finishes
            logger.info_plain_rank0(VISUAL_SEPARATOR)
            
            if use_chat_template or base_chat:
                history.append({"role": ROLE_ASSISTANT, "content": response})

    def _build_base_chat_prompt(self, history):
        """Serialize dialogue as ordinary text for a base completion model."""
        lines = [f"System: {getattr(self.args, 'base_system_prompt', 'You are a helpful assistant.')}"]
        for message in history:
            role = "User" if message["role"] == "user" else "Assistant"
            lines.append(f'{role}: {message["content"]}')
        lines.append("Assistant:")
        return "\n".join(lines)

    @staticmethod
    def _trim_base_chat_response(response):
        """Remove a role continuation generated after the assistant answer."""
        stop_markers = ("\nUser:", "\nSystem:", "\nAssistant:")
        stop_positions = [response.find(marker) for marker in stop_markers if marker in response]
        if stop_positions:
            response = response[:min(stop_positions)]
        return response.strip()

    def _get_sync_input(self):
        """
        In a multi-card environment, only Rank 0 can receive keyboard input.
        The input must be broadcast to other Ranks to ensure all Ranks receive 
        the exact same Prompt for synchronous collective communication.
        """
        if self.rank == 0:
            try:
                user_input = input("\nUser: ").strip()
            except EOFError:
                user_input = "exit"
        else:
            user_input = None

        if dist.is_initialized():
            objects = [user_input]
            dist.broadcast_object_list(objects, src=0)
            user_input = objects[0]
        
        return user_input
