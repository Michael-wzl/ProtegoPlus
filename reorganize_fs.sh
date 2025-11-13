cd face_db/face_scrub && mkdir _noise_db
mv actor_faces/* _noise_db/ && mv actress_faces/* _noise_db/
rm -rf actor_faces actress_faces
protectess=("Bradley_Cooper" "Bruce_Willis" "Christina_Applegate" "Courteney_Cox" "Debra_Messing" "Felicity_Huffman" "Fran_Drescher" "Geena_Davis" "Hugh_Grant" "Jon_Voight" "Jonah_Hill" "Julianna_Margulies" "Julie_Benz" "Kim_Cattrall" "Kristin_Chenoweth" "Lisa_Kudrow" "Matthew_Perry" "Michael_Weatherly" "Sarah_Hyland" "Sarah_Michelle_Gellar")
for name in "${protectess[@]}"; do
    mv _noise_db/"$name" .
done
cd ../..