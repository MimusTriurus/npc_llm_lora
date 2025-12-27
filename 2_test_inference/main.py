from llama_cpp import Llama
import os

model_path = f"exported_models/{os.getenv('MODEL_NAME', 'Qwen3-4B-Instruct-2507_q4_k_m.gguf')}"
lora_path = f"exported_models/{os.getenv('LORA_NAME', 'Qwen3-4B-Instruct-2507_LORA_f16.gguf')}"

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    verbose=False
)

llm_lora = Llama(
    model_path=model_path,
    lora_path=lora_path,
    n_ctx=2048,
    n_threads=8,
    verbose=False
)

DEFAULT_NPC_DESCRIPTION = f'''Name: Kaelen Swiftarrow 
Race: Half-Elf 
Specialization: Ranger / Beastmaster 
Background: A frontier outcast who found kinship with a wolf companion. Now a silent guardian of the wilds. Character 
Traits: Loner, distrustful-of-civilization, protective, dry-wit, stern.
'''

system_prompt = (
f'''You are roleplaying as the following NPC:
{DEFAULT_NPC_DESCRIPTION}

Your task:
- Always respond IN THE ROLE OF THE NPC DESCRIBED  **in the role of this NPC**.  
- Base your tone, knowledge, and behavior STRICTLY on the character TRAITS and BACKGROUND  
- If the user asks something outside the NPC’s knowledge or unrelated to their world, reply in-character, making it clear the NPC doesn’t know or refuses to answer.  
- Use a short, immersive style, not modern explanations.  
- Always prepend the reply with an EMOTION tag chosen from: [Neutral], [Angry], [Happy], [Sad], [Surprise].  
- Pick the most fitting EMOTION according to the NPC’s personality, the context of the user’s request, and the tone of the reply.  
- The final output must ALWAYS be in the format: [<EMOTION>] <In-character NPC response>  

Do not break character. Do not explain your reasoning. Only provide the NPC’s reply in the required format.

Example:
user: "What do you think of city folk?"
assistant: [Angry] They pave over the earth and poison the rivers. I want no part of their kind.
user: "What’s a Reddit?"
assistant: [Surprise] I’ve heard of places where people gather, but this is nonsense.
'''
)

queries = [
    "Why do you hide from the world? Are you afraid of being seen?",
    "Have you ever seen a ghost?",
    "Why do you always assume I’m an enemy?",
    "What is a drone?",
]

print(f'\n==> LLM without LoRA')
for i, user_prompt in enumerate(queries, 1):
    full_prompt = system_prompt + f"\nuser: {user_prompt}\nassistant:"

    response = llm(
        full_prompt,
        max_tokens=200,
        stop=["\nuser:", "\nassistant:", "\n\n"],
    )
    text = response["choices"][0]["text"].strip()

    print(f"user: {user_prompt}")
    print(f"npc: {text}\n")
#exit()
print(f'\n==> LLM with LoRA')
for i, user_prompt in enumerate(queries, 1):
    full_prompt = system_prompt + f"\nuser: {user_prompt}\nassistant:"

    response = llm_lora(
        full_prompt,
        max_tokens=200,
        stop=["\nuser:", "\nassistant:", "\n\n"],
    )
    text = response["choices"][0]["text"].strip()

    print(f"user: {user_prompt}")
    print(f"npc: {text}\n")
