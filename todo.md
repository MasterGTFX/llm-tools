1.call reviewer and git commit (make sure untracked files are added)
2.move all prompts to a prompts/ dir which contains string templates in python for given tool like prompts/edit.py and it contains all user prompts tempates (with {}). this should be standarized in such a way so that we can edit this later like llm_edit.prompts.edit_prompt="abc"
