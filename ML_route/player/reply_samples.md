# LFM2.5 1.2B reply samples

Every line of 10 random MELD test dialogues (seed 11, 4 to 14 lines each), answered twice by `serve_reply.py` through the local player server: once from the line alone, and once with the dialogue's earlier lines as a plain transcript. Each reply answers MELD's own emotion label for the line. The prompt is the sitcom-friend system prompt with two examples, greedy decoding and a 30-token cap.

| | This line only | With the dialogue so far |
|---|---:|---:|
| Replies | 92 | 92 |
| Start with "Hey" | 50 (54%) | 62 (67%) |
| Hit the 30-token cap | 4 (4%) | 16 (17%) |
| Median words | 12 | 15 |
| Median model ms (GTX 1070 Ti, fp16) | 1137 | 1385 |

## Dialogue 77

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Ross:** Okay, where in the hell did you get that?! | surprise | Hey, I was just wondering where the secret sauce went! Want to help me track it down? | Hey, I was just wondering where the secret sauce went! Want to help me track it down? |
| 1 | **Monica:** When Mom and Dad drove you to the hospital to get your nose fixed, I swam into the lake and fished it out. | neutral | That's a cool story, Monica! It sounds like you had a superhero moment in the lake. | Hey Monica, sounds like you had a superhero adventure! 🎣✨ |
| 2 | **Ross:** That cup is mine! | anger | Oh no, Ross, you just claimed the cup belongs to you! Let's settle this with a friendly showdown. | Whoa, you really think you own that cup? Let me guess—your mom’s coffee mug is just a prop? |
| 3 | **Monica:** No it's not! You want it, you're gonna have to win it! | anger | Okay, let's just say I'm not backing down from a challenge—unless it's a nap time! | Hey, I didn’t fish it out—I just swam to the lake and found it! You’re really testing my patience! |
| 4 | **Rachel:** All right, so are we not having dinner at all? | sadness | Rachel, it looks like dinner might be on the menu again! 😅 | Oh no, I was just wondering if we could skip dinner and go straight to the fun stuff! |

## Dialogue 78

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Rachel:** Why? | sadness | Why? | Why? |
| 1 | **Monica:** Because I don't work at the Szechwan Dragon. | neutral | Because I don't work at the Szechwan Dragon. | Yeah, I just realized I didn't work there. It's a nice break! |
| 2 | **Rachel:** You really, really need to get some sleep, honey. | neutral | Hey Rachel, sounds like you're ready for a nap—just don't blame me if you wake up with a story! | Hey Rachel, thanks for the reminder—sleep is the real superpower now! |
| 3 | **Monica:** I know I do. | neutral | Monica, it's good to hear you're in the zone! Let's keep it smooth and smoothie-fueled. | Hey Monica, sounds like you're finally catching up on sleep! 🌙 |
| 4 | **Ross:** Hi. | neutral | Hey Ross, nice to see you! How's your day going so far? | Hey Ross, it's nice to chat! Just remember to keep those sleep hours in mind. |
| 5 | **Rachel:** Hi. | neutral | Hey Rachel, nice to hear your voice! How's your day going so far? | Hey Rachel, it's nice to see you! Just a quick note—remember to prioritize sleep, or we'll have a sleep-deprived |
| 6 | **Ross:** Guess what? | neutral | Hey Ross, guess what? I just remembered we talked about that thing earlier! | Hey Ross, sounds like you're ready to chat about something interesting! |
| 7 | **Ross:** They published my paper. | joy | That's awesome, Ross! Your paper just blew up—congrats! | Hey Ross, sounds like your paper just got a promotion! 🎉 |
| 8 | **Rachel:** Oh, really, let me see, let me see. | joy | Hey Rachel, I'm glad you're here! What's on your mind? | Hey Rachel, looks like you're getting the hang of this! Want to check out the paper together? |
| 9 | **Phoebe:** Rach, look! Oh, hi! Where is my strong Ross Skywalker to come rescue me. There he is. | joy | Hey Phoebe, it looks like Ross is on a mission to rescue you—probably with a side of snacks! | Hey Phoebe, guess what? Ross just dropped a paper and I’m here to help! |

## Dialogue 189

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Joey:** All right! | joy | Hey Joey, sounds like you just won the lottery—time to celebrate! | Hey Joey, sounds like you just won the lottery—time to celebrate! |
| 1 | **Joey:** Westminster Abbey! | joy | Joey, you just said "Westminster Abbey!" – sounds like you're planning a very serious historical tour! | Oh, you just mentioned Westminster Abbey—what a grand spot! Let's hope it's not just a fancy name for a fancy coffee shop! |
| 2 | **Joey:** Hands down, best Abbey I've ever seen. | joy | → "Awesome, you're a real gem! | Hey Joey, you're definitely one of a kind—Westminster Abbey is the real MVP! 🏰✨ |
| 3 | **Joey:** Hey! | joy | Hey Joey, it's great to hear you're feeling up! What's the good news? | Hey Joey, you're welcome! Westminster Abbey is definitely a top pick—just don't forget to bring your own camera! |
| 4 | **Joey:** Okay. | neutral | -> "Got it, let's keep it chill! | Hey Joey, looks like you just got a high-five from the Abbey itself! |
| 5 | **Joey:** What do you think of the Abbey, Chandler? | neutral | What do you think of the Abbey, Chandler? | Hey, that's a great question! The Abbey's got history, but I'm more into the views from the outside. What do you think? |
| 6 | **Chandler:** I think it's great. It's great. Y'know, they're thinking of changing the name of this place. | neutral | Hey, it's awesome! I'm all in for the name change—just make sure it's not "Chandler" again! | Great idea, Chandler! Just make sure they don't rename it to 'The Abbey of Confusion' next. |
| 7 | **Joey:** Really? To what? | surprise | Really? To what? | Wait, what? You're asking about the Abbey? Like, the one with the fancy architecture and the history? 😄 |
| 8 | **Chandler:** To Put the Camera Away!!! | anger | Hey Chandler, if you're gonna yell, at least do it with some context! Maybe we can talk about it instead? | Hey Chandler, if they’re thinking of changing the name, maybe they should just stick with the Abbey and let the history shine! |
| 9 | **Joey:** Man, | neutral | Man, it's just a normal day. | Hey Joey, it sounds like you're vibing with the Abbey vibes—maybe we should start a podcast about it! |

## Dialogue 193

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Ross:** Yes, I made it! | joy | Great news, Ross! Looks like you're officially on the scene! 🎉 | Great news, Ross! Looks like you're officially on the scene! 🎉 |
| 1 | **Ross:** I'm on time! | joy | Hey Ross, great to hear you're on time! Let's make it a great day! | That's awesome, Ross! You really nailed it! |
| 2 | **Ross:** Okay, why don't we all uh, open our books to page 23. | neutral | Hey Ross, sounds like you're ready to dive in—page 23 it is! | Hey Ross, it's good to see you! Let's get those books open and maybe we can all share a laugh about page 23. |
| 3 | **Ross:** Where… Where you will see…a uh…a bunch of uh…red spots. | neutral | Hey Ross, sounds like you're spotting some interesting things—maybe it's just your imagination! | Hey Ross, looks like you found something interesting—maybe it's just a coincidence or a really cool red spot display! |
| 4 | **Ross:** Okay, umm, why don't, why don't you all start to read, while I— | neutral | Hey Ross, it's good to hear you're in the mood for some reading! | Hey Ross, it looks like you're just here for the red spots and not the reading! |

## Dialogue 195

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Chandler:** Okay look, don't...don't be mad okay? | fear | Hey, it's okay, Chandler! Just remember, even superheroes need a little help sometimes. | Hey, it's okay, Chandler! Just remember, even superheroes need a little help sometimes. |
| 1 | **Chandler:** But after I unpacked the boxes I wanted to do something nice for you, so, I-I-I cleaned the apartment. | neutral | Chandler, you really went the extra mile—cleaning the place is the new high score! | Hey Chandler, sounds like you did the whole "clean the place and then be nice" thing—best effort, but maybe next time bring snacks |
| 2 | **Chandler:** So I moved everything around and then I forgot where it, where it went back and I'm sorry, I'm very sorry, I'm sorry, I'm sorry. | sadness | Hey Chandler, it's totally okay to mess things up—apologies are like confetti after a party! | Hey, it's totally understandable to feel that way after a big move—like, you're trying to fix everything but still missing the big picture! |
| 3 | **Monica:** It's okay. Chandler, are you afraid of me or something? | neutral | No worries, Chandler! I'm just being honest—your sandwich was a mystery. | Hey, it's totally normal to feel that way—apologies are good, but maybe next time just text me instead of rushing in |
| 4 | **Chandler:** Do you want me to be afraid of you? | neutral | Hey, I'm just here to chat—no need to scare me! What's on your mind? | Hey, it’s totally fine—just don’t let the fridge start judging you! |
| 5 | **Monica:** Of course not. | neutral | Of course not. | Hey Monica, it’s totally fine—just don’t let the boxes steal your heart again! 😄 |
| 6 | **Monica:** I mean gosh, Chandler what you did, it's, it's a wonderful thing and I really appreciate it. | joy | Hey Monica, that's awesome! Glad you're feeling good about it! | Hey Monica, that sounds like a super cool gesture—maybe you should start a new hobby! |
| 7 | **Monica:** I know I have this weird thing where I want everything to be in the perfect place, but I'd never expect you to worry about that. | joy | Hey Monica, that's a cool vibe! Just remember, even the best planners need a little chaos sometimes. | Hey Monica, sounds like you're trying to be the superhero of organization—congrats on that! 🦸‍♀️ |
| 8 | **Chandler:** Really? | surprise | Hey, you really didn't expect that, did you? Let me know if you need a miracle! | Hey, you really just asked if I'm afraid of you? That's new! But hey, if you need help with anything, I'm |
| 9 | **Monica:** Of course! | neutral | Great! Let's keep the vibe light and positive. | Hey Monica, sounds like you're trying to be the hero again—just don’t forget to hand me the remote next time! |
| 10 | **Chandler:** Oh well you're the best. You come here to me. | joy | Hey Chandler, thanks for the compliment—your smile is the best kind of compliment! | Hey Chandler, that's the spirit—you're the reason I'm still here! 🎉 |
| 11 | **Monica:** All right, hold on okay? First thing's first. Okay, now did Ross sit | neutral | Monica, it looks like Ross is probably still in the kitchen, probably making a sandwich. | Hey Monica, sounds like you're trying to be the calm one—just don't let Ross hear you saying that! |

## Dialogue 206

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Joey:** Man that was great! Huh? Can you believe how long we threw that ball around? | joy | Hey Joey, it was awesome! You were totally the star of the game! | Hey Joey, it was awesome! You were totally the star of the game! |
| 1 | **Rachel:** Yeah, it is amazing it lasted that long. | neutral | That's awesome, Rachel! You really nailed it! | Absolutely, it was a real workout for the ball and the ballpark! |
| 2 | **Ross:** I know. My arm is killing me. | neutral | Hey Ross, sounds like you're dealing with a serious arm situation—let's get you some relief! | Thanks, I guess I just need to stretch more. |
| 3 | **Rachel:** No, I meant with the dropper over here. | neutral | Rachel, you're right—let's focus on the dropper, not the drama! | Oh, I see—so you were just reminding me about the dropper? That’s good to know! |
| 4 | **Chandler:** Y'know, how did I get this reputation as a dropper? Okay? I'm | anger | Hey, it's not your fault the coffee machine is judging you! Let's fix that together! | Hey Chandler, you sure you didn't drop the ball of your life? Let me guess—your arm just quit? |

## Dialogue 208

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Joey:** Hey. | neutral | → "Sounds like you're just here for the snacks! | → "Sounds like you're just here for the snacks! |
| 1 | **Joey:** I just saw a woman breast feeding both of her twins at the same time. it is like a freak show up here. | neutral | Wow, that's pretty cool! Maybe she's a super mom! | Hey Joey, that's pretty wild—maybe you should take a photo for the science fair! |
| 2 | **Joey:** What's the matter? | neutral | I'm just here for the snacks and the quiet vibes. | Hey Joey, it's just a fun observation—maybe you should start a new hobby! |
| 3 | **Rachel:** Nothing. | neutral | Rachel, it looks like you're just here for the quiet vibes. 😊 | Just saying, it's a rare sight, but hey, at least it's not a robot taking over the world. |
| 4 | **Joey:** What is it? Hey! | sadness | Hey Joey, it's okay—just remember, even sad jokes have a punchline! | Hey Joey, it sounds like you're feeling a bit down about that scene. Want to talk about it? |
| 5 | **Rachel:** Really it's nothing. I'm just | sadness | Hey Rachel, it's okay—just don't let it go to your head! | Hey Rachel, it's okay, just a little surprised by the scene! |
| 6 | **Joey:** Rach come on, what? | neutral | Rach come on, what? | Hey Joey, it sounds like you're just curious—maybe you want to know more about the situation? |
| 7 | **Rachel:** I've just been thinking about how my baby and I are gonna be all alone. | sadness | Hey Rachel, it's totally normal to feel that way—just remember, even lonely parents have the best stories! | Hey Rachel, it's totally normal to feel that way—just remember, even the most dramatic mom moments are just part of the parenting story! |
| 8 | **Joey:** What are you talking about alone? What about Ross? | surprise | Wait, what? You're not telling me about Ross? | Hey, it seems like you're asking about Ross—maybe you're wondering if he's still around? 😄 |
| 9 | **Rachel:** Oh please, he'll be with his real family, the twins and little miss new boobs. | sadness | Hey Rachel, let's just say I'm not sure if that's a good plan or a bad one. | Hey Rachel, let's not turn this into a soap opera—just thinking about how cool it is you've got twins and a future! |
| 10 | **Joey:** Okay, how long was I watching that woman? | neutral | Okay, how long was I watching that woman? | Hey Joey, it sounds like you were just processing a lot of information—maybe you need a break from the visuals! |
| 11 | **Rachel:** I'm just saying that y'know, someday Ross is gonna meet somebody and…he's gonna have his own life. Right? | sadness | Hey Rachel, that's a sweet thought—but I hope Ross finds someone who makes him smile, not just someone who makes him think about his past. | Hey Rachel, it sounds like you're trying to be thoughtful about the future, but maybe take a step back before you start planning a life without anyone |
| 12 | **Joey:** Yeah, I guess so. | neutral | → "Sounds like you're just vibing with the vibe! | Hey Joey, sounds like you're just vibing with the vibe—maybe you should start a support group for people who see twins on |
| 13 | **Rachel:** I just never thought I would raise this baby all by myself. Pretty dumb huh? | sadness | Hey Rachel, it's totally normal to feel that way—raising a kid is like a superhero mission, but sometimes the cape gets tangled | Hey Rachel, it's totally normal to feel that way—raising a baby is like a superhero mission, but sometimes the mission involves a lot of |

## Dialogue 224

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Hypnosis Tape:** You are falling fast asleep. | neutral | You are falling fast asleep. | You are falling fast asleep. |
| 1 | **Hypnosis Tape:** Deeper. | neutral | The Hypnosis Tape is sounding calm, so let's keep it that way! | Sure thing! I'm all in for the deep dive. Let's get that mind in the zone! |
| 2 | **Hypnosis Tape:** Deeper. | neutral | The Hypnosis Tape is sounding calm, so let's keep it that way! | Sounds calm, just letting the waves wash over you like a superhero in a bathtub. |
| 3 | **Hypnosis Tape:** Deeper. | neutral | The Hypnosis Tape is sounding calm, so let's keep it that way! | Sounds calm, just letting the waves wash over you like a superhero in a bathtub. |
| 4 | **Hypnosis Tape:** You are now completely asleep. | neutral | Sounds great, let's drift off into a peaceful zone! | Sounds like you're in a deep nap—time to drift off even faster! |
| 5 | **Hypnosis Tape:** You don't need to smoke. | neutral | Sounds great! You're feeling so relaxed and smoke-free! | Sounds great, let's keep the smoke-free zone alive! |
| 6 | **Hypnosis Tape:** Cigarettes don't control you. | neutral | The Hypnosis Tape is sounding calm and saying, "That's a great reminder—let's keep it that way! | Sounds like you're already in a deep nap—just let the calm vibes keep you going! |
| 7 | **Hypnosis Tape:** You are a strong, confident woman, who does not need to smoke. | neutral | That's a great vibe! Let's keep that calm, confident energy going. | I’m feeling really strong and confident, like I’m the boss of my own life! |
| 8 | **Hypnosis Tape:** A strong, confident woman. | neutral | The Hypnosis Tape is sounding calm and steady, like a soothing lullaby for your mind. | Hey there, I'm just here to keep things calm and confident! |

## Dialogue 231

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Chandler:** I know. | neutral | Hey Chandler, just let me know when you're ready to chat! | Hey Chandler, just let me know when you're ready to chat! |
| 1 | **Chandler:** See, yes. | neutral | Hey Chandler, sounds like you're all set—just keep it up! | That's good to hear, Chandler! Let's keep the momentum going. |
| 2 | **Chandler:** That's Yasmine Bleeth, she's a completely different kind of chick. | neutral | Hey, I just thought you were the cool one—maybe we should start a debate club! | That's a nice observation, Chandler! You're definitely picking up on the vibe. |
| 3 | **Chandler:** I love you both. | neutral | Hey Chandler, sounds like you're feeling pretty good about both of them! | Hey, that's sweet of you to say—maybe we should start a fan club for Yasmine and Chandler! |
| 4 | **Chandler:** But in very different ways. | neutral | Hey Chandler, I'm just saying your vibe is totally different—like a mood swing in a jazz band! | Hey, that's a good point—different vibes, same energy! |
| 5 | **Joey:** Hey! | surprise | Hey Joey, did you just realize I was the one who was totally surprised? | Hey Joey, you just mentioned Yasmine Bleeth—sounds like you're trying to impress someone with your knowledge! |
| 6 | **Chandler:** Hey! | surprise | Hey Chandler, did you hear that? I thought you were still in the kitchen! | Hey, that's a surprise! What's the deal? |
| 7 | **Chandler:** I did! But the store wouldn't take her back! So then I took her to the shelter, and you know what I found out? | anger | Hey, it's not your fault the store was all "no" and the shelter had a better vibe! | Oh no, Chandler, you just made me feel like I’m the only one who gets taken for granted! |
| 8 | **Chandler:** If they can't find a home for her, they kill her! And I'm not gonna let that happen to little Yasmine! | anger | Hey, if you're going to go all out, at least make sure Yasmine gets a cozy spot instead of a sad ending! | Oh no, Chandler, you just made the shelter look like a dumpster fire! Let's find Yasmine a better place! |
| 9 | **Joey:** Okay, good, good, good, 'cause, good, 'cause I was kinda having second thoughts too. | fear | Hey Joey, it's totally normal to have second thoughts—just don't let them steal the spotlight! | Hey Joey, don’t worry—your heart’s as big as a pizza, but your mind’s got a little too much drama. Let’s |
| 10 | **Chandler:** Okay. And it's not just chicks y'know? It's all kinds of other animals! | disgust | Hey, if it's not just chicks, then what's next—bugs in the house?! | Oh no, Chandler, you just made the shelter look like a dumpster fire! Let's hope Yasmine finds a better home than a trash |
| 11 | **Joey:** That's horrible! Well, you did the right thing man. | disgust | Guess you're saying I'm the hero now? Let's see if you can keep it up! | Hey Joey, if you really care about animals, maybe think twice before taking Yasmine to the shelter—unless you want to become a hero |
| 12 | **Chandler:** Thanks, I'm glad you see it that way. | joy | Hey Chandler, great to hear you're feeling good about it! Let's keep the good vibes going! | Hey Chandler, sounds like you just saved the day—go shine like a superhero with a snack! |
| 13 | **Chandler:** Ohhh-hoo, funny story! | joy | Hey Chandler, you just made me laugh so hard I thought I might burst into a stand-up routine! | Oh, I love when Chandler gets that laugh—like a puppy seeing a laser pointer! |

## Dialogue 249

| # | Line | Label | This line only | With the dialogue so far |
|---:|---|---|---|---|
| 0 | **Mr. Kaplan:** Eh. | joy | Hey there, Mr. Kaplan! Sounds like you're just vibing—no need to rush! | Hey there, Mr. Kaplan! Sounds like you're just vibing—no need to rush! |
| 1 | **Rachel:** Oh, you got me. | neutral | Hey Rachel, that's a good vibe! Let's keep the conversation light. | Hey Rachel, that's a good one—let's just say I'm intrigued! |
| 2 | **Mr. Kaplan:** Well, don't think I haven't noticed your potential. | neutral | Hey there, Mr. Kaplan! Just checking in—did I miss anything important? | Hey, that's a nice compliment—maybe I should start a confidence-building seminar! |
| 3 | **Mr. Kaplan:** Well, I've got a project for you that's a lot more related to fashion. | neutral | Mr. Kaplan, it's great to hear you're diving into fashion! Let's make some stylish plans! | Hey, that sounds interesting—maybe we can blend fashion with fashion again! |
| 4 | **Mr. Kaplan:** How does that sound? | neutral | That's a good question, Mr. Kaplan. Let's break it down clearly. | That sounds interesting, Mr. Kaplan! Let's see if we can make it trendy. |
| 5 | **Rachel:** Oh, that sounds great. | joy | Hey Rachel, that sounds great! Let's make it happen. | That sounds great, Rachel! Let's make it stylish! |
| 6 | **Mr. Kaplan:** Come on over here, sweetheart. | neutral | Hey there, Mr. Kaplan! Sounds like you're ready to chat—just don't expect me to cry! | Hey there, Mr. Kaplan! Sounds like you're ready to chat about fashion projects—let's make it stylish! |
| 7 | **Rachel:** Oh thank you so much Mr. Kaplan, thank you so much. | joy | Hey Rachel, that's the spirit! Let's keep the good vibes going! | Hey Rachel, you're welcome! Let's get that fashion project started—no judgment, just fabulous ideas! |
