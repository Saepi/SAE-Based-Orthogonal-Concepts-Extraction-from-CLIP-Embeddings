python step3_concept_naming/step3_concept_naming.py \
    --sae_checkpoint model/sae_final_models/sae_l11e-7_dim4096-epoch=199.ckpt \
    --vocab_txt data/vocab/clipdissect_20k.txt \
    --vocab_emb data/vocab/embeddings_clip_RN50_clipdissect_20k.pth \
    --output_dir step3_concept_naming/concept_names \
    --output_csv concept_names_noort.csv \
    --device cuda

python step3_concept_naming/step3_concept_naming.py \
    --sae_checkpoint model/sae_final_models/sae_ortsae_l1e-6-epoch=199.ckpt \
    --vocab_txt data/vocab/clipdissect_20k.txt \
    --vocab_emb data/vocab/embeddings_clip_RN50_clipdissect_20k.pth \
    --output_dir step3_concept_naming/concept_names \
    --output_csv concept_names_ortsae.csv \
    --device cuda

python step3_concept_naming/step3_concept_naming.py \
    --sae_checkpoint model/sae_final_models/sae_final_frobenius_5e-7-epoch=175.ckpt \
    --vocab_txt data/vocab/clipdissect_20k.txt \
    --vocab_emb data/vocab/embeddings_clip_RN50_clipdissect_20k.pth \
    --output_dir step3_concept_naming/concept_names \
    --output_csv concept_names_frob.csv \
    --device cuda

python step3_concept_naming/step3_concept_naming.py \
    --sae_checkpoint model/sae_final_models/sae_srip_l1e-7_lr1e-4-epoch=195.ckpt \
    --vocab_txt data/vocab/clipdissect_20k.txt \
    --vocab_emb data/vocab/embeddings_clip_RN50_clipdissect_20k.pth \
    --output_dir step3_concept_naming/concept_names \
    --output_csv concept_names_srip.csv \
    --device cuda