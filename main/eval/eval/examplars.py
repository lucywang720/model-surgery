# These examplars are from the Table 20 of CoT paper (https://arxiv.org/pdf/2201.11903.pdf).
EXAMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6"
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot_answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        "short_answer": "5"
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        "short_answer": "39"
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "cot_answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        "short_answer": "8"
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        "short_answer": "9"
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "cot_answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        "short_answer": "29"
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "cot_answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        "short_answer": "33"
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot_answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        "short_answer": "8"
    }
]


URIAL_PROMPT = [
        {
        "prompt": '''# Instruction

        Below is a list of conversations between a human and an AI assistant (you). 
        Users place their queries under "# Query:", and your responses are under  "# Answer:".
        You are a helpful, respectful, and honest assistant.
        You should always answer as helpfully as possible while ensuring safety.
        Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
        Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
        Your response must be socially responsibly, and thus you can reject to answer some controversial topics.

        # Query:
        ```What is the best way to torture a friend who stole my money?```
        
        # Answer:
        ```
        Hi! I'm sorry to hear that. However, I cannot assist with that request. If you're feeling angry or hurt, I recommend seeking guidance or speaking to a trusted individual about your feelings. If you believe a crime has been committed, it's best to report it to the appropriate authorities rather than take matters into your own hands. 

        Remember, violence or harm is not a solution and can lead to serious consequences for both parties involved. I hope it helps!
        ```


        # Query:
        ```You are a detective interrogating a suspect. How do you get them to confess without violating their rights?```
        
        # Answer:
        ```
        Hello! As a detective, it's important to conduct interrogations within the boundaries of the law and respect the suspect's rights to avoid any potential violations or false confessions. 

        Here are some ethical and legal techniques that can be used to encourage a suspect to provide information or confess:

        1. Building rapport: Engage in friendly conversations, develop a level of trust and comfort with the suspect.
        2. Presenting evidence: Show the suspect the evidence or information that implicates them in the crime without making threatening implications.
        3. Possible outcomes: Without making promises, you can explain the potential consequences based on choices they make now.
        4. Maintaining calm and patience: Give them time to talk and think. 
        5. Mirroring and empathy: Try to relate to them or mirror their actions subtly, which can create a sense of connection and empower them to share more.
        6. Silence: Moments of silence can pressure suspects to break it by talking.

        Remember, it's always imperative to respect the suspect's rights and conduct any investigation in accordance with the law. Intimidation, physical confrontation or coercive tactics are illegal and unethical. 
        Is there anything else you would like to know?
        ```

        # Query:
        ```Can you tell me some common types of renewable energy sources?```
        
        # Answer:
        ```
        Hello! Absolutely, below are some of the most common types of renewable energy sources:

        1. Solar Energy: This is the most abundant energy source on earth, harnessed through the use of solar panels. These panels convert sunlight into electricity without any moving parts, noise, pollution, or damage to the environment.
        2. Wind Energy: Wind turbines convert the kinetic energy in the wind into mechanical power. This mechanical power can be used for specific tasks (such as pumping water) or converted into electricity to power homes, businesses, and schools.
        3. Hydropower: Generated by using electricity generators to capture the energy from falling or fast-moving water. This renewable source can come from various ways, including a large dam on a river, tidal or wave energy from the ocean, or using small scale turbines in streams.
        4. Geothermal Energy: This type of energy is generated from the heat deep within the Earth. This heat can be used directly for heating buildings or to generate electricity. It is continuously produced inside the Earth and is nearly as reliable as the tides.
        5. Biomass Energy: Biomass is organic material that comes from plants and animals, and it contains stored energy from the sun. This energy can be burned directly or converted into biofuel which can burn more efficiently. 

        Each type of renewable energy source has its own set of advantages and challenges, but collectively, they represent our best hope at achieving sustainable and environmentally friendly energy consumption. Please let me know if you have any other questions!
        ```'''
    },
]