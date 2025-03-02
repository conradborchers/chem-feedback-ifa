library(tidyverse)
library(lme4)

d_ref <- read.csv('Feedback zugeordnet - Tabellenblatt3.csv', header=FALSE) %>%
  tibble() %>%
  `colnames<-`(c('feedback_text', 'feedback_category')) %>%
  mutate(feedback_category = tolower(feedback_category)) %>%
  mutate(feedback_category = ifelse(feedback_category=='-', NA, feedback_category)) %>%
  mutate(feedback_category = ifelse(str_detect(feedback_category, 'correctness'), 'correctness_feedback', feedback_category))

d_pk <- read_csv('Auswertung - scores.csv') %>%
  rename(user_id = user) %>%
  mutate(high_pk = ifelse(overall > median(overall, na.rm=TRUE), 'high', 'low'))

d_log <- read_csv('Cleaned Log Data - Blatt 1 - log-data-precourse-st.csv', skip = 1) %>%
  mutate(feedback_text = coalesce(feedback_text...25, feedback_text...54)) %>%
  mutate(kc_default = ifelse(tutor=='ORCCA', 'ORCCA', kc_default)) %>%
  filter(!is.na(kc_default)) %>%
  filter(attempt_at_step == 1) %>%
  filter(outcome %in% c('CORRECT', 'INCORRECT', 'HINT')) %>%
  mutate(outcome = ifelse(outcome == 'CORRECT', 1, 0)) %>%
  mutate(feedback_text = ifelse(is.na(feedback_text), '', feedback_text)) %>%
  select(user_id, tutor, time, problem_name, step_name, kc_default, feedback_text, outcome) %>%
  fuzzyjoin::stringdist_left_join(d_ref, by = "feedback_text", max_dist = 2, method = "osa") %>%
  select(-matches('feedback_text')) %>%
  mutate(feedback_category = ifelse(is.na(feedback_category), 'None', feedback_category))

d_afm <- d_log %>%
  arrange(user_id, time) %>%
  mutate(prior_feedback = lag(feedback_category, 1)) %>%
  mutate(prior_feedback = ifelse(user_id != lag(user_id, 1), 'None', prior_feedback)) %>% 
  select(user_id, tutor, time, kc_default, prior_feedback, outcome) %>%
  group_by(user_id, kc_default) %>%
  mutate(opportunity = 1:n()) %>%
  ungroup() %>%
  left_join(d_pk %>% select(user_id, high_pk), by='user_id') %>%
  mutate(high_pk = ifelse(is.na(high_pk), 'missing', high_pk))

m <- glmer(outcome ~ (1 | user_id) + (1 | kc_default) + opportunity, d_afm %>% filter(tutor=='Stoich'), family='binomial', nAGQ=0, verbose=2)
sjPlot::tab_model(m)

m <- glmer(outcome ~ (1 | user_id) + (1 | kc_default) + opportunity, d_afm %>% filter(tutor=='Stoich') %>%
             filter(high_pk=='low'), family='binomial', nAGQ=0, verbose=2)
sjPlot::tab_model(m)

# Hard to model performance in ORCCA without a skill model
m <- glmer(outcome ~ (1 | user_id)  + opportunity, d_afm %>% filter(tutor!='Stoich'), family='binomial', nAGQ=0, verbose=2)
sjPlot::tab_model(m)

d_ifa <- d_afm %>%
  filter(high_pk=='low') %>% # tweak this for low vs. high prior knowledge analysis
  filter(tutor=='Stoich') %>%
  group_by(user_id, kc_default, prior_feedback) %>%
  arrange(time) %>%
  mutate(opportunity = row_number()) %>%
  select(user_id, time, kc_default, prior_feedback, opportunity, outcome) %>%
  pivot_wider(names_from = prior_feedback, values_from = opportunity, values_fn = first) %>%
  arrange(user_id, time) %>%
  janitor::clean_names() %>%
  group_by(user_id, kc_default) %>%
  fill(everything(), .direction = "down") %>%  # Fill all columns
  ungroup() %>%
  mutate(across(where(is.numeric), ~replace_na(.x, 0)))  # Replace NA with 0 for numeric columns

names(d_ifa)
d_ifa %>%
  select(
    explicit_correction,
      correctness_feedback,
      indirect_feedback,
      metalinguistic_feedback,
      elicitation
  )

m_ifa <- glmer(outcome ~ (1 | user_id) + (1 | kc_default) + 
                 explicit_correction + 
                 correctness_feedback + 
                 indirect_feedback + 
                 metalinguistic_feedback + 
                 elicitation, d_ifa, family='binomial', nAGQ=0, verbose = 2)
sjPlot::tab_model(m_ifa)
